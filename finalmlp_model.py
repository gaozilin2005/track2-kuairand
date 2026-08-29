"""FinalMLP（AAAI 2023）：两路 MLP，各自对同一份 embedding 做不同的门控加权，
再用多头双线性融合两路输出。跟 DeepFM 不一样，这个架构**完全不含**显式交互项——
没有 FM 的双线性项，纯靠两路 MLP + 门控 + 双线性融合的组合，测的是一个跟 FM
不同的归纳偏置，不是在 FM 上叠加。torch 实现（本文件是唯一用到这个架构的地方）。

架构（跟论文一致）：
  e        = 7 个域 embedding 拼接                              (B, 7k)
  h1 = 2·σ(gate1(e)) ⊙ e                                        (B, 7k)   门控 1（MMOE 风格）
  h2 = 2·σ(gate2(e)) ⊙ e                                        (B, 7k)   门控 2，独立参数，给两路不同的加权视角
  o1 = MLP_1(h1)                                                 (B, d)
  o2 = MLP_2(h2)                                                 (B, d)
  z  = 多头双线性融合(o1, o2) + b                                 (B,)      把 o1/o2 切成 H 个头，每头一个双线性矩阵

跟 deepfm_model.py 一样刻意保持小（两路 MLP 各 2 层，64→64，dropout 0.2）——
这个数据集只有 114 万行，本项目已经反复验证"更大不等于更好"。

训练用 BPR，采样逻辑跟 baseline.py 的 run_fm(loss='pairwise') 一致。对照基线
是纯 BPR FM 7 域（0.6008），跟 DeepFM 用同一个对照基准，直接可比。
"""
import argparse, collections, copy, time
import numpy as np
import torch
import torch.nn as nn

from data import load, encode, FIELDS
from evaluate import evaluate
from sequence_model import resolve_device


class FinalMLP(nn.Module):
    def __init__(self, dim, n_fields, k=16, stream_dim=64, n_heads=4, dropout=0.2):
        super().__init__()
        assert stream_dim % n_heads == 0, "stream_dim 必须整除 n_heads"
        self.n_heads = n_heads
        self.head_dim = stream_dim // n_heads
        e_dim = n_fields * k

        self.V = nn.Embedding(dim, k)
        nn.init.normal_(self.V.weight, 0, 0.01)

        self.gate1 = nn.Linear(e_dim, e_dim)
        self.gate2 = nn.Linear(e_dim, e_dim)

        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.PReLU(), nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim), nn.PReLU(),
            )
        self.stream1 = mlp(e_dim, stream_dim)
        self.stream2 = mlp(e_dim, stream_dim)

        self.bilinear_W = nn.Parameter(torch.zeros(n_heads, self.head_dim, self.head_dim))
        nn.init.normal_(self.bilinear_W, 0, 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, X):
        e = self.V(X).flatten(1)                          # (B, 7k)

        h1 = 2 * torch.sigmoid(self.gate1(e)) * e
        h2 = 2 * torch.sigmoid(self.gate2(e)) * e
        o1 = self.stream1(h1)                              # (B, d)
        o2 = self.stream2(h2)                              # (B, d)

        B = o1.shape[0]
        o1h = o1.view(B, self.n_heads, self.head_dim)       # (B, H, d/H)
        o2h = o2.view(B, self.n_heads, self.head_dim)
        # 每个头一个双线性矩阵：score_h = o1_h^T W_h o2_h，H 个头求和。
        score = torch.einsum('bhi,hij,bhj->bh', o1h, self.bilinear_W, o2h).sum(1)
        return self.b + score


def _predict(model, X, device, bs=65_536):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).long().to(device)
            outs.append(model(xb).cpu().numpy())
    model.train()
    return np.concatenate(outs)


def run_finalmlp(splits, k=16, stream_dim=64, n_heads=4, dropout=0.2, lr=0.001, epochs=40,
                  bs=8192, patience=4, seed=0, device='auto', verbose=True, return_scores=False):
    device = resolve_device(device) if isinstance(device, str) else device
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    torch.manual_seed(seed)
    model = FinalMLP(dim, n_fields=Xtr.shape[1], k=k, stream_dim=stream_dim,
                      n_heads=n_heads, dropout=dropout).to(device)
    if verbose:
        print(f"  device: {device} | fields: {Xtr.shape[1]} | stream_dim: {stream_dim} | heads: {n_heads}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(seed)

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
    ap.add_argument('--stream_dim', type=int, default=64)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_finalmlp(splits, k=a.k, stream_dim=a.stream_dim, n_heads=a.n_heads,
                        dropout=a.dropout, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device)
    print(f"\n=== finalmlp (seed={a.seed}, stream_dim={a.stream_dim}, heads={a.n_heads}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
