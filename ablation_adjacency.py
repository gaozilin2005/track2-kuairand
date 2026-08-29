"""第三个便宜诊断：author_recency 分桶后发现收益主要来自"同一 session 内紧邻"这个
近乎确定性的效应（gap≈0 时 long_view 率 99.95%），1 小时开外基本是平的，不是 DIEN
假设的平滑衰减。这里单独测这个更简单的版本："这一行是不是紧接在同一用户对同一
作者的一次 long_view 之后"——纯粹是相邻关系，不分桶、不需要 author_recency 的
分位数机制。看它单独能不能接住 author_recency (+0.0017) 的大部分收益。

跟另外两个 ablation_*.py 用同一套评测口径：单独作为第 6 域接到原始 5 域 FM 上，
BPR，5 seed，与 BPR FM 的 0.5971 做严格对照（不是加在 7 域新默认之上）。
"""
import statistics
import numpy as np
from data import load, build_vocab, FIELDS
from evaluate import evaluate
import baseline as B

D = './KuaiRand-Pure/data'
splits = load(D)
print({k: len(v) for k, v in splits.items()})

_DATE, _USER, _VIDEO, _AUTHOR, _TAB, _DUR, _LABEL, _TIME = range(8)
split_names = list(splits.keys())

all_user, all_time, all_author, all_lv, all_split, all_pos = [], [], [], [], [], []
for si, name in enumerate(split_names):
    for pos, x in enumerate(splits[name]):
        all_user.append(x[_USER]); all_time.append(x[_TIME])
        all_author.append(x[_AUTHOR]); all_lv.append(x[_LABEL])
        all_split.append(si); all_pos.append(pos)

all_time = np.asarray(all_time, dtype=np.int64)
_, user_int = np.unique(all_user, return_inverse=True)
_, author_int = np.unique(all_author, return_inverse=True)

# 只按 (user, time) 排序——不像 author_recency 那样先按 author 分组，
# 因为这里要的是"这个用户时间线上紧挨着的前一条"，不分作者。
order = np.lexsort((all_time, user_int))
u_sorted = user_int[order]; a_sorted = author_int[order]
lv_sorted = np.asarray(all_lv)[order]
split_sorted = np.asarray(all_split)[order]; pos_sorted = np.asarray(all_pos)[order]

feat = {name: np.zeros(len(splits[name]), dtype=np.int32) for name in split_names}
prev_author = None; prev_lv = False; prev_user = None
for i in range(len(order)):
    u = u_sorted[i]
    if u == prev_user and prev_lv and a_sorted[i] == prev_author:
        feat[split_names[split_sorted[i]]][pos_sorted[i]] = 1
    prev_user = u; prev_author = a_sorted[i]; prev_lv = bool(lv_sorted[i])

n_hit = sum(int(v.sum()) for v in feat.values())
total = len(order)
print(f"adjacency=1 的行数: {n_hit} / {total} ({100*n_hit/total:.3f}%)")

# 复用 data.py 的 5 域词表，追加第 6 域：adjacency ∈ {0,1}，不需要 UNK 槽。
vocabs, unk, field_dims, offsets, edges = build_vocab(splits)
dim5 = int(sum(field_dims))
_raw = lambda x: [x[_USER], x[_VIDEO], x[_AUTHOR], x[_TAB], str(int(np.searchsorted(edges, x[_DUR])))]

enc = {}
for name, rws in splits.items():
    X = np.empty((len(rws), 6), dtype=np.int32)
    y = np.empty(len(rws), dtype=np.float32)
    users = []
    for n, x in enumerate(rws):
        for i, v in enumerate(_raw(x)):
            X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
        X[n, 5] = dim5 + feat[name][n]
        y[n] = x[_LABEL]
        users.append(x[_USER])
    enc[name] = (X, y, users)
dim = dim5 + 2

Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

import collections
def run_bpr(seed, epochs=40, bs=8192, patience=4):
    m = B.FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
    for i, (u, yv) in enumerate(zip(utr, ytr)):
        (user_pos if yv > 0 else user_neg)[u].append(i)
    mixed = [u for u in user_pos if u in user_neg]
    pos_blocks = [np.array(user_pos[u]) for u in mixed]
    neg_pools = [np.array(user_neg[u]) for u in mixed]
    pos_idx_all = np.concatenate(pos_blocks)
    counts = np.array([len(b) for b in pos_blocks])

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                       for pool, c in zip(neg_pools, counts)])
        perm = rng.permutation(len(pos_idx_all))
        pi, ni = pos_idx_all[perm], neg_idx_all[perm]
        for i in range(0, len(pi), bs):
            m.step_pairwise(Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]])
        va = evaluate(uva, yva, m.predict(Xva))['primary']
        if va > best + 1e-5:
            best, bad = va, 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return evaluate(ute, yte, m.predict(Xte))

results = [run_bpr(seed) for seed in range(5)]
for seed, r in enumerate(results):
    print(f"  seed {seed}: GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")

primary = [r['primary'] for r in results]
gauc = [r['GAUC'] for r in results]
ndcg = [r['nDCG@5'] for r in results]
print(f"\n=== 6-field BPR FM (+adjacency), 5 seeds ===")
for name, vals in [('GAUC', gauc), ('nDCG@5', ndcg), ('primary', primary)]:
    print(f"  {name:8s} mean={statistics.mean(vals):.4f} std={statistics.pstdev(vals):.4f}")
print("  vs. BPR FM (5-field, RUN_LOG.md): GAUC 0.6638 | nDCG@5 0.5304 | primary 0.5971")
print("  vs. BPR FM + prior_exposure:      GAUC 0.6662 | nDCG@5 0.5310 | primary 0.5986")
print("  vs. BPR FM + author_recency:      GAUC 0.6663 | nDCG@5 0.5313 | primary 0.5988")
