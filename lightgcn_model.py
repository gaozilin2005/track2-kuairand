"""LightGCN（He et al., SIGIR 2020）：在 user-item 二部图上传播 embedding，
读**多跳协同信号**——这是 FM 结构上拿不到的信息。

为什么值得试（依据本项目自己的实验结论，不是泛泛的"换个模型"）：
异构集成那条记录里，DeepFM 和 FinalMLP 的预测相关性高达 **0.973**——架构看着天差地别，
其实都在用不同机器算同一个 `user_id × video_id` 交互，所以对集成毫无贡献。真正有贡献的是
BST（相关性 0.885~0.892），因为它**读了不同的信息**（顺序）。

LightGCN 属于后者：文献对这个区别说得很直白——矩阵分解/FM 只看到图的**一阶边**
（用户和物品直接连的那条边），而图卷积把 user→item→user→item 这样的多跳路径也聚合进来。
同一批交互数据，但提取的是拓扑结构而非独立样本。所以它有理由跟 FM 系不相关。

架构（严格照 LightGCN 论文，不加料）：
  A_hat = D^-1/2 A D^-1/2         对称归一化邻接矩阵，A 来自 train 里的 long_view=1 边
  E^(k+1) = A_hat @ E^(k)         纯传播：**没有**特征变换，**没有**非线性激活
  E_final = mean(E^(0..K))        各层平均
  score(u,i) = E_final[u] · E_final[i]
论文的核心主张就是把 NGCF 里的变换和激活全删掉反而更好——本项目也反复验证过
"加容量没用"，所以这个精简版正合适。

训练用 BPR，采样逻辑跟 baseline.py 的 run_fm(loss='pairwise') 一致。
注意：LightGCN 只用 user/video 两个 ID，不吃 author/tab/dur_bucket/时序特征——
它单独的分数大概率打不过 7 域 FM，**但这不是重点**：它是作为集成成员存在的，
判断标准是"错得跟别人不一样吗"，不是"单独分数高不高"（BST 那条记录的教训）。
"""
import argparse, collections, copy, time
import numpy as np
import torch
import torch.nn as nn

from data import load, encode, FIELDS
from evaluate import evaluate
from sequence_model import resolve_device


def build_graph(splits, device):
    """从 train 的 long_view=1 交互建对称归一化邻接矩阵（torch sparse）。
    user/item 的索引空间覆盖所有 split（避免 valid/test 里出现越界 id），
    但**边只来自 train**——不能用评测期的交互建图，那是泄漏。"""
    all_rows = splits['train'] + splits['valid'] + splits['test']
    uids = sorted({x[1] for x in all_rows})
    vids = sorted({x[2] for x in all_rows})
    u_idx = {u: i for i, u in enumerate(uids)}
    v_idx = {v: i for i, v in enumerate(vids)}
    n_u, n_v = len(uids), len(vids)

    src, dst = [], []
    for x in splits['train']:
        if x[6]:                                  # 只连 long_view=1 的边
            src.append(u_idx[x[1]]); dst.append(n_u + v_idx[x[2]])
    src = np.asarray(src); dst = np.asarray(dst)
    print(f"  graph: {n_u} users, {n_v} items, {len(src)} positive edges from train")

    rows = np.concatenate([src, dst])             # 无向图：两个方向都加
    cols = np.concatenate([dst, src])
    n = n_u + n_v
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    deg[deg == 0] = 1.0
    vals = (1.0 / np.sqrt(deg[rows]) / np.sqrt(deg[cols])).astype(np.float32)

    A = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])).long(),
        torch.from_numpy(vals), (n, n)).coalesce().to(device)
    return A, u_idx, v_idx, n_u, n_v


class LightGCN(nn.Module):
    def __init__(self, n_u, n_v, A, k=64, n_layers=3):
        super().__init__()
        self.n_u, self.A, self.n_layers = n_u, A, n_layers
        self.E = nn.Embedding(n_u + n_v, k)
        nn.init.normal_(self.E.weight, 0, 0.01)

    def propagate(self):
        """返回各层平均后的最终 embedding。每次前向都要重算（图卷积没有缓存）。"""
        e = self.E.weight
        acc = e
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self.A, e)
            acc = acc + e
        return acc / (self.n_layers + 1)

    def score(self, emb, u, v):
        return (emb[u] * emb[self.n_u + v]).sum(-1)


def _predict(model, u_arr, v_arr, device, bs=200_000):
    model.eval()
    outs = []
    with torch.no_grad():
        emb = model.propagate()
        for i in range(0, len(u_arr), bs):
            u = torch.from_numpy(u_arr[i:i + bs]).long().to(device)
            v = torch.from_numpy(v_arr[i:i + bs]).long().to(device)
            outs.append(model.score(emb, u, v).cpu().numpy())
    model.train()
    return np.concatenate(outs)


def run_lightgcn(splits, k=64, n_layers=3, lr=0.001, epochs=40, bs=8192, patience=4,
                 seed=0, device='auto', verbose=True, return_scores=False):
    device = resolve_device(device) if isinstance(device, str) else device
    A, u_idx, v_idx, n_u, n_v = build_graph(splits, device)

    def to_arrays(name):
        rws = splits[name]
        return (np.array([u_idx[x[1]] for x in rws], dtype=np.int64),
                np.array([v_idx[x[2]] for x in rws], dtype=np.int64),
                np.array([x[6] for x in rws], dtype=np.float32),
                [x[1] for x in rws])
    utr, vtr, ytr, users_tr = to_arrays('train')
    uva, vva, yva, users_va = to_arrays('valid')
    ute, vte, yte, users_te = to_arrays('test')

    torch.manual_seed(seed)
    model = LightGCN(n_u, n_v, A, k=k, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"  device: {device} | k={k} | layers={n_layers}")

    # 与 baseline.py 完全相同的正负样本池（同用户曝光内配对）。
    user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
    for i, (u, yv) in enumerate(zip(users_tr, ytr)):
        (user_pos if yv > 0 else user_neg)[u].append(i)
    mixed = [u for u in user_pos if u in user_neg]
    pos_blocks = [np.array(user_pos[u]) for u in mixed]
    neg_pools = [np.array(user_neg[u]) for u in mixed]
    pos_idx_all = np.concatenate(pos_blocks)
    counts = np.array([len(b) for b in pos_blocks])
    if verbose:
        print(f"  pairwise: {len(mixed)} mixed users, {len(pos_idx_all)} pairs/epoch")

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        neg_idx = np.concatenate([rng.choice(p, size=c, replace=True)
                                   for p, c in zip(neg_pools, counts)])
        perm = rng.permutation(len(pos_idx_all))
        pi, ni = pos_idx_all[perm], neg_idx[perm]

        losses = []
        for i in range(0, len(pi), bs):
            bpi, bni = pi[i:i + bs], ni[i:i + bs]
            emb = model.propagate()               # 图卷积每个 batch 重算一次
            up = torch.from_numpy(utr[bpi]).long().to(device)
            vp = torch.from_numpy(vtr[bpi]).long().to(device)
            vn = torch.from_numpy(vtr[bni]).long().to(device)
            un = torch.from_numpy(utr[bni]).long().to(device)
            zp = model.score(emb, up, vp)
            zn = model.score(emb, un, vn)
            loss = -torch.log(torch.sigmoid(zp - zn) + 1e-9).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        va = evaluate(users_va, yva, _predict(model, uva, vva, device))
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
    sva = _predict(model, uva, vva, device)
    ste = _predict(model, ute, vte, device)
    if return_scores:
        return {'valid_scores': sva, 'test_scores': ste,
                'valid': evaluate(users_va, yva, sva), 'test': evaluate(users_te, yte, ste)}
    return {'valid': evaluate(users_va, yva, sva), 'test': evaluate(users_te, yte, ste)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()})
    res = run_lightgcn(splits, k=a.k, n_layers=a.n_layers, lr=a.lr,
                       epochs=a.epochs, patience=a.patience, seed=a.seed, device=a.device)
    print(f"\n=== lightgcn (seed={a.seed}, k={a.k}, layers={a.n_layers}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
