"""DeepFM：在已验证的 FM 交互项上加一个并行的 DNN 分支，torch 实现。

架构：z = z_FM + z_DNN。z_FM 跟 baseline.py 的 FM 完全同构（双线性交互，7 域，
共享同一张 embedding 表 V）；z_DNN 是把 7 个域的 embedding 拼起来（7k 维）过一个
小 MLP，捕捉 FM 的双线性形式在结构上完全表达不出来的非线性/高阶组合。两路加法
合并——如果 DNN 什么都没学到，z_FM 这条已验证的路径完全不受影响（跟 DIN 的
e_interest、watchtime 辅助任务用的是同一种"加法接入，不动已验证部分"的思路）。

刻意保持 DNN 很小（默认两层，64→32，dropout 0.2）：这个数据集只有 114 万行，
本项目已经反复验证"更大不等于更好"（FM 本身的 embedding 维度消融、这一路 DeepFM
实验本身要测的也正是"换结构"而非"堆参数"），大 DNN 在这个规模上过拟合风险是真实的。

训练用 BPR，采样逻辑跟 baseline.py 的 run_fm(loss='pairwise') 完全一致，只是梯度交给
torch.autograd。对照基线是纯 BPR FM 7 域（0.6008，RUN_LOG.md）——不叠 watchtime
辅助任务，先干净地测 DNN 分支本身有没有用。
"""
import argparse, collections, copy, time
import numpy as np
import torch
import torch.nn as nn

from data import load, encode, FIELDS
from evaluate import evaluate
from sequence_model import resolve_device


class DeepFM(nn.Module):
    def __init__(self, dim, n_fields, k=16, hidden=(64, 32), dropout=0.2):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        self.W = nn.Embedding(dim, 1)
        self.b = nn.Parameter(torch.zeros(()))

        layers = []
        in_dim = n_fields * k
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.dnn = nn.Sequential(*layers)

        nn.init.normal_(self.V.weight, 0, 0.01)
        nn.init.zeros_(self.W.weight)

    def forward(self, X):
        E = self.V(X)                                   # (B,F,k)
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W(X).squeeze(-1).sum(1)
        z_fm = self.b + lin + inter

        z_dnn = self.dnn(E.flatten(1)).squeeze(-1)        # (B,) 7 域 embedding 拼接过 MLP
        return z_fm + z_dnn


def _predict(model, X, device, bs=65_536):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).long().to(device)
            outs.append(model(xb).cpu().numpy())
    model.train()
    return np.concatenate(outs)


def run_deepfm(splits, k=16, hidden=(64, 32), dropout=0.2, lr=0.001, epochs=40, bs=8192,
               patience=4, seed=0, device='auto', verbose=True, return_scores=False):
    device = resolve_device(device) if isinstance(device, str) else device
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    torch.manual_seed(seed)
    model = DeepFM(dim, n_fields=Xtr.shape[1], k=k, hidden=hidden, dropout=dropout).to(device)
    if verbose:
        print(f"  device: {device} | fields: {Xtr.shape[1]} | dnn hidden: {hidden}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(seed)

    # 跟 baseline.py 的 run_fm(loss='pairwise') 完全相同的正负样本池构造。
    user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
    for i, (u, yv) in enumerate(zip(utr, ytr)):
        (user_pos if yv > 0 else user_neg)[u].append(i)
    mixed = [u for u in user_pos if u in user_neg]
    pos_blocks = [np.array(user_pos[u]) for u in mixed]
    neg_pools = [np.array(user_neg[u]) for u in mixed]
    pos_idx_all = np.concatenate(pos_blocks)
    counts = np.array([len(b) for b in pos_blocks])
    if verbose:
        print(f"  pairwise: {len(mixed)} mixed users, {len(pos_idx_all)} pos-anchored pairs/epoch")

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                       for pool, c in zip(neg_pools, counts)])
        perm = rng.permutation(len(pos_idx_all))
        pi, ni = pos_idx_all[perm], neg_idx_all[perm]

        losses = []
        for i in range(0, len(pi), bs):
            xp = torch.from_numpy(Xtr[pi[i:i + bs]]).long().to(device)
            xn = torch.from_numpy(Xtr[ni[i:i + bs]]).long().to(device)
            zp, zn = model(xp), model(xn)
            loss = -torch.log(torch.sigmoid(zp - zn) + 1e-9).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, _predict(model, Xva, device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    sva, ste = _predict(model, Xva, device), _predict(model, Xte, device)
    if return_scores:
        return {'valid_scores': sva, 'test_scores': ste,
                'valid': evaluate(uva, yva, sva), 'test': evaluate(ute, yte, ste)}
    return {'valid': evaluate(uva, yva, sva), 'test': evaluate(ute, yte, ste)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--hidden', type=int, nargs='+', default=[64, 32])
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_deepfm(splits, k=a.k, hidden=tuple(a.hidden), dropout=a.dropout, lr=a.lr,
                      epochs=a.epochs, seed=a.seed, device=a.device)
    print(f"\n=== deepfm (seed={a.seed}, hidden={a.hidden}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
