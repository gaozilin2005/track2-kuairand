"""全库检索任务的 baseline 阶梯（KuaiRand-Pure，正例 = is_click，指标 NDCG@10 / Recall@50）。

跟 baseline.py 的关系：**不同任务，不是同一套东西的改版。** baseline.py 做用户内排序
（对该用户的曝光重排，GAUC/nDCG@5，正例 long_view）；这里做全库检索（从 7,583 个视频里
给每个用户排序）。两边分数没有可比性。

任务变化带来的两个实质影响：
1. **负例来源变了。** baseline.py 的 BPR 从"该用户自己的曝光"里采负例——因为那个任务
   只需要在曝光内部排序。检索任务要在全库排序，负例必须从**整个视频库**采，否则模型
   永远没学过"没被曝光的视频该排多低"。
2. **不能用曝光上下文特征。** 全库检索要给"从未曝光过的 (user, video) 对"打分，而
   `tab` 这类特征是曝光时的上下文，对没发生过的曝光根本没有取值。所以模型只能用
   user × item 和物品侧属性——这也是为什么这里用 MF/矩阵分解，而不是 baseline.py 的 7 域 FM。

  --model random : 随机打分（下界，自检评测代码）
  --model pop    : 按点击量排序（trivial baseline，检索任务里这个通常不弱）
  --model mf     : BPR-MF，全库均匀采负例
"""
import argparse, collections, time
import numpy as np
from data import load
from evaluate_retrieval import evaluate_retrieval, build_eval_sets

CLICK = 8          # data.load() 行元组里 is_click 的位置


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def build_index(splits):
    """用全部 split 的视频建索引（候选集 = 视频库全集）。"""
    vids = sorted({x[2] for rows in splits.values() for x in rows})
    return {v: i for i, v in enumerate(vids)}


def run_random(splits, item_index, eval_split, seed=0):
    users, user_pos, exclude = build_eval_sets(splits, eval_split, item_index)
    rng = np.random.default_rng(seed)
    n_items = len(item_index)
    S = rng.random((len(users), n_items)).astype(np.float32)
    return evaluate_retrieval(S, user_pos, exclude)


def run_pop(splits, item_index, eval_split):
    users, user_pos, exclude = build_eval_sets(splits, eval_split, item_index)
    n_items = len(item_index)
    cnt = np.zeros(n_items, dtype=np.float32)
    for x in splits['train']:
        if x[CLICK]:
            i = item_index.get(x[2])
            if i is not None:
                cnt[i] += 1
    S = np.broadcast_to(cnt, (len(users), n_items))
    return evaluate_retrieval(lambda sl: np.array(S[sl], dtype=np.float32), user_pos, exclude)


def run_mf(splits, item_index, eval_split, k=64, lr=0.01, l2=1e-5, epochs=30,
           bs=8192, patience=3, seed=0, verbose=True):
    """BPR-MF：正例是 train 里 is_click=1 的 (user, item)，负例从全库均匀采。"""
    n_items = len(item_index)
    train_users = sorted({x[1] for x in splits['train']})
    all_users = sorted({x[1] for rows in splits.values() for x in rows})
    uidx = {u: i for i, u in enumerate(all_users)}
    n_users = len(all_users)

    pu, pi = [], []
    seen = collections.defaultdict(set)
    for x in splits['train']:
        i = item_index.get(x[2])
        if i is None:
            continue
        seen[uidx[x[1]]].add(i)
        if x[CLICK]:
            pu.append(uidx[x[1]]); pi.append(i)
    pu = np.asarray(pu, dtype=np.int64); pi = np.asarray(pi, dtype=np.int64)
    if verbose:
        print(f"  {n_users} users, {n_items} items, {len(pu)} positive (click) train pairs")

    rng = np.random.default_rng(seed)
    P = (rng.normal(0, 0.01, (n_users, k))).astype(np.float32)
    Q = (rng.normal(0, 0.01, (n_items, k))).astype(np.float32)
    mP, vP = np.zeros_like(P), np.zeros_like(P)
    mQ, vQ = np.zeros_like(Q), np.zeros_like(Q)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0

    users_va, pos_va, excl_va = build_eval_sets(splits, 'valid', item_index)
    va_rows = np.array([uidx[u] for u in users_va], dtype=np.int64)

    def score_fn(P_, Q_, rows):
        def f(sl):
            return (P_[rows[sl]] @ Q_.T).astype(np.float32)
        return f

    best, best_state, bad = -1e9, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(len(pu))
        losses = []
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            u, ip = pu[idx], pi[idx]
            inn = rng.integers(0, n_items, size=len(idx))       # 全库均匀负采样
            Pu, Qp, Qn = P[u], Q[ip], Q[inn]
            diff = np.einsum('ij,ij->i', Pu, Qp - Qn)
            g = (sigmoid(diff) - 1.0).astype(np.float32)
            B = len(idx)
            gP = (g[:, None] * (Qp - Qn)) / B
            gQp = (g[:, None] * Pu) / B
            gQn = (-g[:, None] * Pu) / B
            dP = np.zeros_like(P); dQ = np.zeros_like(Q)
            np.add.at(dP, u, gP)
            np.add.at(dQ, ip, gQp)
            np.add.at(dQ, inn, gQn)
            dP += l2 * P; dQ += l2 * Q
            t += 1
            for M, V_, D, W in ((mP, vP, dP, P), (mQ, vQ, dQ, Q)):
                M *= b1; M += (1 - b1) * D
                V_ *= b2; V_ += (1 - b2) * (D * D)
                W -= lr * (M / (1 - b1 ** t)) / (np.sqrt(V_ / (1 - b2 ** t)) + eps)
            losses.append(float(-np.mean(np.log(sigmoid(diff) + 1e-9))))

        r = evaluate_retrieval(score_fn(P, Q, va_rows), pos_va, excl_va)
        cur = r['NDCG@10']
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid NDCG@10 {r['NDCG@10']:.4f} "
                  f"Recall@50 {r['Recall@50']:.4f} | {time.time()-t0:.1f}s")
        if cur > best + 1e-6:
            best, bad, best_state = cur, 0, (P.copy(), Q.copy())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    P, Q = best_state

    users_e, pos_e, excl_e = build_eval_sets(splits, eval_split, item_index)
    rows_e = np.array([uidx[u] for u in users_e], dtype=np.int64)
    return evaluate_retrieval(score_fn(P, Q, rows_e), pos_e, excl_e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--model', default='mf', choices=['random', 'pop', 'mf'])
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    item_index = build_index(splits)
    print({k: len(v) for k, v in splits.items()}, f"catalog={len(item_index)} videos")

    if a.model == 'random':
        res = run_random(splits, item_index, a.split, seed=a.seed)
    elif a.model == 'pop':
        res = run_pop(splits, item_index, a.split)
    else:
        res = run_mf(splits, item_index, a.split, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)

    print(f"\n=== {a.model} on {a.split} (retrieval, click=positive) ===")
    print(f"  NDCG@10   {res['NDCG@10']:.4f}")
    print(f"  Recall@50 {res['Recall@50']:.4f}")
    print(f"  (NDCG@50 {res['NDCG@50']:.4f} | Recall@10 {res['Recall@10']:.4f} | users scored {res['users']})")
