"""大规模 bonus 数据集（1K / 27K）用的 FM + BPR，torch 实现，稀疏 embedding 更新。

为什么需要这个（不是重复造轮子）：`run_bonus.py` 里的 numpy FM 复用了 `baseline.py`
的 `FM._adam_update`，那个函数对**整张 embedding 表**做稠密 Adam 更新——每个 minibatch
不管实际碰到多少行，都要做一次 O(dim × k) 的运算。Pure 的 dim=40,273 无所谓；
1K 的 dim=5,778,436 已经让每个 epoch 从 Pure 的 2-3 秒变成 316-416 秒（`RUN_LOG.md`
2026-08-30 记录，已识别未修）；27K 的 dim 预计约 3500 万，同样的做法在任何硬件上
都不现实——这不是内存问题，是纯粹浪费的计算量。

修法：`nn.Embedding(sparse=True)` + `torch.optim.SparseAdam`——只更新一个 batch 里
真正查过的行，不去碰整张表。这是工业界处理千万级 embedding 表的标准做法，
本项目也早就在用 torch（`sequence_model.py` / `deepfm_model.py` / `lightgcn_model.py`），
这里只是把同样的工具用在"表大到数值 Adam 划不来"这个新出现的场景。

架构跟 `baseline.py` 的 FM 完全同构（双线性交互 + 一阶项 + bias），BPR 训练，
数值上应该给出跟 numpy 版一致（在随机性范围内）的结果——这是一个实现方式的改变，
不是新模型。

数据仍走 `data_large.py` 的列式格式（内存高效），只是训练循环换成 GPU + 稀疏更新。
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn

from evaluate import evaluate


class SparseFM(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.V = nn.Embedding(dim, k, sparse=True)
        self.W = nn.Embedding(dim, 1, sparse=True)
        self.b = nn.Parameter(torch.zeros(()))          # bias 是标量，稠密更新没有代价
        nn.init.normal_(self.V.weight, 0, 0.01)
        nn.init.zeros_(self.W.weight)

    def forward(self, X):
        E = self.V(X)                                    # (B,F,k)
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W(X).squeeze(-1).sum(1)
        return self.b + lin + inter


def _predict(model, X, device, bs=200_000):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).long().to(device)
            outs.append(model(xb).cpu().numpy())
    model.train()
    return np.concatenate(outs)


def run(enc, dim, k=16, lr=0.01, epochs=40, bs=8192, patience=4, seed=0,
        device='auto', verbose=True):
    import collections
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if device == 'auto' else torch.device(device)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    torch.manual_seed(seed)
    model = SparseFM(dim, k=k).to(device)
    # bias 是普通稠密参数，稀疏参数（V, W）用 SparseAdam，两组分开建 optimizer。
    opt_sparse = torch.optim.SparseAdam(list(model.V.parameters()) + list(model.W.parameters()), lr=lr)
    opt_dense = torch.optim.Adam([model.b], lr=lr)
    rng = np.random.default_rng(seed)
    if verbose:
        print(f'  device: {device} | dim={dim} | k={k}')

    user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
    for i, (u, yv) in enumerate(zip(utr, ytr)):
        (user_pos if yv > 0 else user_neg)[u].append(i)
    mixed = [u for u in user_pos if u in user_neg]
    pos_blocks = [np.array(user_pos[u], dtype=np.int64) for u in mixed]
    neg_pools = [np.array(user_neg[u], dtype=np.int64) for u in mixed]
    user_pos.clear(); user_neg.clear()
    pos_idx_all = np.concatenate(pos_blocks)
    counts = np.array([len(b) for b in pos_blocks])
    if verbose:
        print(f'  {len(mixed)} mixed users, {len(pos_idx_all)} pos-anchored pairs/epoch')

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        neg_idx = np.concatenate([rng.choice(p, size=c, replace=True)
                                   for p, c in zip(neg_pools, counts)])
        perm = rng.permutation(len(pos_idx_all))
        pi, ni = pos_idx_all[perm], neg_idx[perm]

        losses = []
        for i in range(0, len(pi), bs):
            xp = torch.from_numpy(Xtr[pi[i:i + bs]]).long().to(device)
            xn = torch.from_numpy(Xtr[ni[i:i + bs]]).long().to(device)
            zp, zn = model(xp), model(xn)
            loss = -torch.log(torch.sigmoid(zp - zn) + 1e-9).mean()
            opt_sparse.zero_grad(); opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step(); opt_dense.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, _predict(model, Xva, device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f'  early stop at epoch {ep}')
                break

    model.load_state_dict(best_state)
    return {'valid': evaluate(uva, yva, _predict(model, Xva, device)),
            'test': evaluate(ute, yte, _predict(model, Xte, device)),
            'valid_best': best}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='1k', choices=['pure', '1k', '27k'])
    ap.add_argument('--data_dir', default='./KuaiRand-1K/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()

    from data_large import load_columnar, encode_columnar
    import resource, sys
    def peak_gb():
        # ru_maxrss is bytes on macOS/BSD but KILOBYTES on Linux — without this factor
        # the cluster silently under-reports by exactly 1024x (confirmed: an observed
        # "0.04 GB" on the 27K run was actually ~41 GB).
        factor = 1 if sys.platform == 'darwin' else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor / (1024 ** 3)

    t_start = time.time()
    print(f'loading {a.data_dir} (suffix={a.suffix}) ...')
    data, vocab = load_columnar(a.data_dir, suffix=a.suffix)
    print(f'  peak mem after load: {peak_gb():.2f} GB')
    enc, dim = encode_columnar(data, vocab)
    data.clear()
    print(f'  peak mem after encode: {peak_gb():.2f} GB')

    res = run(enc, dim, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device)
    print(f"\n=== KuaiRand-{a.suffix.upper()} : SparseFM + BPR (GPU), seed={a.seed} ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    print(f"  wall-clock {time.time()-t_start:.0f}s | peak mem {peak_gb():.2f} GB")
