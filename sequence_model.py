"""DIN 风格用户历史序列 + FM，torch 实现（本文件是仓库里唯一用 torch 的地方）。

架构：静态 5 域 FM（与 baseline.py 的 FM 完全同构）之外，加一个第 6 个
"伪域"：用户历史（最近 L 个 long_view=1 视频）经 DIN 局部激活单元对候选
video_id 做 attention，得到 e_interest，再按 FM 的双线性恒等式并入交互项：
    inter_full = inter_static + (S · e_interest)
（e_interest 的平方项在展开 (S+e_interest)^2 时与 sum(E^2)+||e_interest||^2
 中的 ||e_interest||^2 项相消，见 RUN_LOG.md 的推导）
外加一个一阶项 w_int · e_interest。因为 e_interest 是对候选 item 做
attention 的结果，在同一用户的曝光组内并不恒定，不会落入"用户侧一阶特征
对组内排序恒为 0"的陷阱（README「从哪里开始改」一节已验证过这一点）。

训练用 BPR（与 baseline.py 的 --loss pairwise 完全相同的采样逻辑），
只是梯度交给 torch.autograd，不再手推。baseline.py 的 numpy FM 保持不动，
作为已经记录在 RUN_LOG.md 里的对照基线。

2026-08-28 修订：第一版（纯 embedding 特征：hist/cand/diff/prod）实测没有
收益（RUN_LOG.md）。诊断实验（ablation_prior_exposure.py）发现原因不是
"没有序列信号"，而是 DIN 的 softmax attention 没能从极稀疏的样本
（训练集里只有 0.18% 的行是精确重复曝光）里学会把权重锐利地压到精确命中
的历史条目上——一个简单的 0/1"是否精确复看过"特征反而比整个 attention
机制更有效。这一版直接把 same_video（该历史条目和候选视频是否是同一
video_id）这个 0/1 标志拼进 DIN 局部激活单元的输入，把这个模式直接喂给
网络，而不是指望它从 diff≈0 这个角落自己学出来。这个修订也没有收益
（RUN_LOG.md）。

2026-08-28 再修订：`prior_exposure`/`author_recency`/`adjacency` 三个诊断
统一指向一个结论——序列信号真实存在，但形状是 session 边界的阶梯效应，
不是平滑衰减；`data.py` 已经把前两个收编为默认特征（0.5971 → 0.6008）。
DIN 的失败具体在于它把历史当无序集合做 attention pooling，从头到尾拿不到
"顺序"这个维度的信息。新增 `BSTFM`（Behavior Sequence Transformer 风格）：
history + 候选拼成一条序列，加可学习的位置编码，过一层标准 Transformer
self-attention（`nn.TransformerEncoderLayer`，双向、非因果，与 BST 论文
一致），取候选自己那个位置的输出作为 e_interest，其余（e_interest 折进
FM 交互项的方式、BPR 训练、PAD 处理的正确性论证）与 DIN 版本完全一致，
只换了"怎么从历史里算出 e_interest"这一步。这是当前测试的直接对照：
既然静态特征已经吃掉了阶梯效应的收益，BST 要证明自己得在那之上——对照
基线是当前最优的 7 域 FM（0.6008），不是被拉平的旧 5 域数字（0.5971）。
"""
import argparse, collections, copy, time
import numpy as np
import torch
import torch.nn as nn

from data import load, encode, build_vocab, FIELDS
from sequence import build_history
from evaluate import evaluate


class SeqFM(nn.Module):
    def __init__(self, dim, k=16, L=160, hidden=32):
        super().__init__()
        self.pad_idx = dim
        self.V = nn.Embedding(dim + 1, k, padding_idx=dim)   # +1 行给 PAD；padding_idx 保证该行梯度恒为 0
        self.W = nn.Embedding(dim, 1)                        # Hist 从不经过 W，不需要 PAD 行
        self.b = nn.Parameter(torch.zeros(()))
        self.w_int = nn.Parameter(torch.zeros(k))
        self.attn = nn.Sequential(nn.Linear(4 * k + 1, hidden), nn.PReLU(), nn.Linear(hidden, 1))

        nn.init.normal_(self.V.weight, 0, 0.01)
        nn.init.zeros_(self.W.weight)
        with torch.no_grad():
            self.V.weight[dim].zero_()   # 显式清零 PAD 行（其梯度已被 padding_idx 机制屏蔽，永远保持为 0）

    def forward(self, X, Hist):
        E = self.V(X)                                  # (B,5,k) 静态域，与 baseline.py 的 FM 同构
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W(X).squeeze(-1).sum(1)
        z0 = self.b + lin + inter

        e_cand = E[:, 1, :]                              # video_id 域自己的 embedding，与历史共享同一张表
        E_hist = self.V(Hist)                            # (B,L,k)；PAD 槽位查出来恒为 0（padding_idx）
        mask = (Hist != self.pad_idx)                    # (B,L)

        cand_b = e_cand.unsqueeze(1).expand_as(E_hist)
        same_video = (Hist == X[:, 1:2]).float().unsqueeze(-1)   # (B,L,1) 显式"精确复看"标志；
                                                                    # PAD 槽位的 Hist==pad_idx 永远 != 真实
                                                                    # video_id，天然是 0，无需特殊处理
        feat = torch.cat([E_hist, cand_b, E_hist - cand_b, E_hist * cand_b, same_video], dim=-1)  # (B,L,4k+1)
        score = self.attn(feat).squeeze(-1)               # (B,L)
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=1) * mask        # 全 PAD 的行 softmax 后是均匀分布在垃圾值上，
                                                            # 乘一次 mask 强制清零，而不是依赖 -1e9 的下溢
        e_interest = (alpha.unsqueeze(-1) * E_hist).sum(1)  # (B,k)

        return z0 + (S * e_interest).sum(1) + e_interest @ self.w_int


class BSTFM(nn.Module):
    """Behavior Sequence Transformer 风格：history + 候选拼成一条 (L+1) 长度的
    序列，加可学习位置编码，过标准 Transformer self-attention（双向、非因果），
    取候选自己那个位置的输出作为 e_interest。折进 FM 交互项的方式与 SeqFM
    完全一致（见模块头注释的推导）。

    PAD 槽位安全性：候选自己的位置永远不被 mask（它不是历史，天然有效），
    所以 self-attention 的 key_padding_mask 里每一行至少有一个有效 key——
    不会出现"整行全被 mask"的退化情况（DIN 版本靠 softmax 后再乘一次 mask
    修的那个 bug，这里从设计上就不会发生）。更进一步：候选位置的输出只从
    "未被 mask 的 key"里加权得到，PAD 槽位的 value 权重恒为 0（softmax 对
    -inf 项的输出恒为 0），所以 e_interest 对 PAD embedding 的取值完全不
    敏感，梯度也就恒为 0——跟 DIN 版本证明的性质一样，只是这次是 Transformer
    自己的 masking 机制保证的，不需要额外手动乘 mask。
    """
    def __init__(self, dim, k=16, L=160, n_heads=4, n_layers=1, ff_dim=64, dropout=0.1):
        super().__init__()
        self.pad_idx = dim
        self.L = L
        self.V = nn.Embedding(dim + 1, k, padding_idx=dim)
        self.W = nn.Embedding(dim, 1)
        self.b = nn.Parameter(torch.zeros(()))
        self.w_int = nn.Parameter(torch.zeros(k))
        self.pos_emb = nn.Embedding(L + 1, k)   # L 个历史位置 + 候选自己的位置
        layer = nn.TransformerEncoderLayer(d_model=k, nhead=n_heads, dim_feedforward=ff_dim,
                                            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        nn.init.normal_(self.V.weight, 0, 0.01)
        nn.init.zeros_(self.W.weight)
        nn.init.normal_(self.pos_emb.weight, 0, 0.01)
        with torch.no_grad():
            self.V.weight[dim].zero_()

    def forward(self, X, Hist):
        E = self.V(X)
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W(X).squeeze(-1).sum(1)
        z0 = self.b + lin + inter

        e_cand = E[:, 1, :]
        E_hist = self.V(Hist)                             # (B,L,k)
        mask = (Hist != self.pad_idx)                      # (B,L) True=真实历史

        seq = torch.cat([E_hist, e_cand.unsqueeze(1)], dim=1)          # (B,L+1,k)：候选拼在最后一个位置
        pos_ids = torch.arange(self.L + 1, device=X.device).unsqueeze(0).expand(X.shape[0], -1)
        seq = seq + self.pos_emb(pos_ids)

        cand_valid = torch.ones(X.shape[0], 1, dtype=torch.bool, device=X.device)
        key_padding_mask = ~torch.cat([mask, cand_valid], dim=1)       # True=该 key 被屏蔽；候选永远不被屏蔽

        out = self.transformer(seq, src_key_padding_mask=key_padding_mask)  # (B,L+1,k)
        e_interest = out[:, -1, :]                          # 只取候选自己那个位置的输出

        return z0 + (S * e_interest).sum(1) + e_interest @ self.w_int


def resolve_device(name='auto'):
    """'auto' -> cuda 优先，否则 cpu。其他字符串（'cuda'/'cpu'/'mps'/'cuda:0'…）原样传给 torch.device。"""
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _predict(model, X, H, device, bs=8192):
    # bs must stay small (unlike baseline.py's plain-FM predict, which can afford 200k):
    # each row's attention MLP materializes a (bs, L, 4k) tensor, so a too-large bs here
    # blows up memory (e.g. L=160 at bs=200k is a >5GB single tensor) instead of speeding
    # anything up. Same bound applies on GPU (just scale bs up if VRAM allows — this default
    # is sized for CPU RAM, not tuned for GPU throughput).
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).long().to(device)
            hb = torch.from_numpy(H[i:i + bs]).long().to(device)
            outs.append(model(xb, hb).cpu().numpy())
    model.train()
    return np.concatenate(outs)


def run_seq(splits, k=16, hidden=32, lr=0.001, epochs=40, bs=8192, patience=4,
            seed=0, L=160, device='auto', arch='din',
            n_heads=4, n_layers=1, ff_dim=64, dropout=0.1, verbose=True, return_scores=False):
    device = resolve_device(device) if isinstance(device, str) else device
    enc, dim = encode(splits)
    vocabs, unk, field_dims, offsets, edges = build_vocab(splits)
    hist = build_history(splits, vocabs, unk, offsets, pad_idx=dim, L=L)

    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    Htr, Hva, Hte = hist['train'], hist['valid'], hist['test']

    torch.manual_seed(seed)
    if arch == 'din':
        model = SeqFM(dim, k=k, L=L, hidden=hidden).to(device)
    elif arch == 'bst':
        model = BSTFM(dim, k=k, L=L, n_heads=n_heads, n_layers=n_layers,
                      ff_dim=ff_dim, dropout=dropout).to(device)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    if verbose:
        print(f"  arch: {arch} | device: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(seed)

    # 与 baseline.py 的 run_fm(loss='pairwise') 完全相同的正负样本池构造。
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
            bpi, bni = pi[i:i + bs], ni[i:i + bs]
            xp = torch.from_numpy(Xtr[bpi]).long().to(device); hp = torch.from_numpy(Htr[bpi]).long().to(device)
            xn = torch.from_numpy(Xtr[bni]).long().to(device); hn = torch.from_numpy(Htr[bni]).long().to(device)
            zp, zn = model(xp, hp), model(xn, hn)
            loss = -torch.log(torch.sigmoid(zp - zn) + 1e-9).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, _predict(model, Xva, Hva, device))
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
    sva, ste = _predict(model, Xva, Hva, device), _predict(model, Xte, Hte, device)
    if return_scores:
        return {'valid_scores': sva, 'test_scores': ste,
                'valid': evaluate(uva, yva, sva), 'test': evaluate(ute, yte, ste)}
    return {'valid': evaluate(uva, yva, sva), 'test': evaluate(ute, yte, ste)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--arch', default='din', choices=['din', 'bst'],
                    help='din=局部激活单元 pooling（无序）/ bst=Transformer self-attention + 位置编码（有序）')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--hidden', type=int, default=32, help='仅 --arch din 用：局部激活单元隐层大小')
    ap.add_argument('--n_heads', type=int, default=4, help='仅 --arch bst 用：多头数，须整除 k')
    ap.add_argument('--n_layers', type=int, default=1, help='仅 --arch bst 用：Transformer 层数')
    ap.add_argument('--ff_dim', type=int, default=64, help='仅 --arch bst 用：feedforward 隐层大小')
    ap.add_argument('--dropout', type=float, default=0.1, help='仅 --arch bst 用')
    ap.add_argument('--L', type=int, default=160)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto',
                    help="'auto'（默认，cuda 优先否则 cpu）/ 'cpu' / 'cuda' / 'mps' / 'cuda:0' 等")
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_seq(splits, k=a.k, hidden=a.hidden, lr=a.lr, epochs=a.epochs, seed=a.seed, L=a.L,
                  device=a.device, arch=a.arch, n_heads=a.n_heads, n_layers=a.n_layers,
                  ff_dim=a.ff_dim, dropout=a.dropout)
    print(f"\n=== seq_fm arch={a.arch} (seed={a.seed}, L={a.L}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
