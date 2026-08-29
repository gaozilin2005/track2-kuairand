"""LightGCN 用于**全库检索**（NDCG@10 / Recall@50，正例 = is_click）。

跟 lightgcn_model.py 的关系：同一个模型，但那个文件是给"用户内排序"任务用的（已实测
没有收益——因为那个任务能用 author/tab/dur_bucket/时序特征，而 LightGCN 只有 user×item，
信息量本来就少）。检索任务反过来：**没法用曝光上下文特征**（要给从未曝光过的 (user,video)
对打分，`tab` 这种曝光时才有的东西根本没有取值），所以 user×item 就是这个任务的自然形式，
LightGCN 在这里是对口的，不是勉强套用。

跟 retrieval_baseline.py 的 MF 的唯一区别就是多了图传播：
  MF        : score(u,i) = P[u] · Q[i]
  LightGCN  : 先在 user-item 二部图上传播 K 轮再取各层平均，然后同样点积
所以两者可以直接对比，差异完全归因于"多跳协同信号"这一项。

图只用 **train 里 is_click=1 的边**建（评测期的交互不能进图，那是泄漏）。
负例从**全库**均匀采样（跟 retrieval_baseline.py 的 MF 一致）。
"""
import argparse, collections, copy, time
import numpy as np
import torch
import torch.nn as nn

from data import load
from evaluate_retrieval import evaluate_retrieval, build_eval_sets
from retrieval_baseline import build_index
from sequence_model import resolve_device

CLICK = 8


class LightGCNRetrieval(nn.Module):
    def __init__(self, n_u, n_v, A, k=64, n_layers=3):
        super().__init__()
        self.n_u, self.A, self.n_layers = n_u, A, n_layers
        self.E = nn.Embedding(n_u + n_v, k)
        nn.init.normal_(self.E.weight, 0, 0.01)

    def propagate(self):
        e = self.E.weight
        acc = e
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self.A, e)
            acc = acc + e
        return acc / (self.n_layers + 1)


def build_click_graph(splits, uidx, item_index, device):
    src, dst = [], []
    n_u = len(uidx)
    for x in splits['train']:
        if x[CLICK]:
            i = item_index.get(x[2])
            if i is not None:
                src.append(uidx[x[1]]); dst.append(n_u + i)
    src = np.asarray(src); dst = np.asarray(dst)
    n = n_u + len(item_index)
    rows = np.concatenate([src, dst]); cols = np.concatenate([dst, src])
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    deg[deg == 0] = 1.0
    vals = (1.0 / np.sqrt(deg[rows]) / np.sqrt(deg[cols])).astype(np.float32)
    A = torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])).long(),
                                torch.from_numpy(vals), (n, n)).coalesce().to(device)
    print(f"  graph: {n_u} users, {len(item_index)} items, {len(src)} click edges from train")
    return A


def run(splits, item_index, eval_split='test', k=64, n_layers=3, lr=0.01, epochs=30,
        bs=8192, patience=3, seed=0, device='auto', verbose=True):
    device = resolve_device(device) if isinstance(device, str) else device
    all_users = sorted({x[1] for rows in splits.values() for x in rows})
    uidx = {u: i for i, u in enumerate(all_users)}
    n_u, n_v = len(all_users), len(item_index)

    A = build_click_graph(splits, uidx, item_index, device)

    pu, pi = [], []
    for x in splits['train']:
        if x[CLICK]:
            i = item_index.get(x[2])
            if i is not None:
                pu.append(uidx[x[1]]); pi.append(i)
    pu = np.asarray(pu, dtype=np.int64); pi = np.asarray(pi, dtype=np.int64)
    if verbose:
        print(f"  device: {device} | k={k} layers={n_layers} | {len(pu)} positive train pairs")

    torch.manual_seed(seed)
    model = LightGCNRetrieval(n_u, n_v, A, k=k, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(seed)

    users_va, pos_va, excl_va = build_eval_sets(splits, 'valid', item_index)
    rows_va = np.array([uidx[u] for u in users_va], dtype=np.int64)

    def make_score_fn(rows):
        model.eval()
        with torch.no_grad():
            emb = model.propagate()
            U = emb[:n_u].cpu().numpy()
            V = emb[n_u:].cpu().numpy()
        model.train()
        return lambda sl: (U[rows[sl]] @ V.T).astype(np.float32)

    best, best_state, bad = -1e9, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(len(pu))
        losses = []
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            u = torch.from_numpy(pu[idx]).long().to(device)
            ip = torch.from_numpy(pi[idx]).long().to(device)
            inn = torch.from_numpy(rng.integers(0, n_v, size=len(idx))).long().to(device)
            emb = model.propagate()
            eu = emb[u]
            zp = (eu * emb[n_u + ip]).sum(-1)
            zn = (eu * emb[n_u + inn]).sum(-1)
            loss = -torch.log(torch.sigmoid(zp - zn) + 1e-9).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        r = evaluate_retrieval(make_score_fn(rows_va), pos_va, excl_va)
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid NDCG@10 {r['NDCG@10']:.4f} "
                  f"Recall@50 {r['Recall@50']:.4f} | {time.time()-t0:.1f}s")
        if r['NDCG@10'] > best + 1e-6:
            best, bad, best_state = r['NDCG@10'], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    users_e, pos_e, excl_e = build_eval_sets(splits, eval_split, item_index)
    rows_e = np.array([uidx[u] for u in users_e], dtype=np.int64)
    return evaluate_retrieval(make_score_fn(rows_e), pos_e, excl_e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    item_index = build_index(splits)
    print({k_: len(v) for k_, v in splits.items()}, f"catalog={len(item_index)}")
    res = run(splits, item_index, a.split, k=a.k, n_layers=a.n_layers, lr=a.lr,
              epochs=a.epochs, seed=a.seed, device=a.device)
    print(f"\n=== lightgcn on {a.split} (retrieval, click=positive) ===")
    print(f"  NDCG@10   {res['NDCG@10']:.4f}")
    print(f"  Recall@50 {res['Recall@50']:.4f}")
    print(f"  (NDCG@50 {res['NDCG@50']:.4f} | Recall@10 {res['Recall@10']:.4f} | users {res['users']})")
