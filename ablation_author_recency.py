"""第二个便宜诊断实验：加一个"距离上次 long_view 这个作者的作品过了多久"的分桶特征。

跟 ablation_prior_exposure.py 的 prior_exposure（"见过这个视频吗"，纯 set membership）
不同，这个特征测的是**时间衰减**——BST/DIEN 在 DIN 之上加的核心东西正是"顺序/新鲜度"，
而不只是"见过没见过"。FM 已经有 user_id × author_id 的静态交叉项了；这个特征测的是
"最近有没有看这个作者"这种随时间变化的新鲜度信号，是否比静态的用户-作者共现携带更多信息。

如果这个也没用：对"该不该投入 DIEN 的 auxiliary loss 机制"是很有分量的负面证据。
如果这个有用：说明时序信号是真实存在的，DIEN 值得投入。

跟 ablation_prior_exposure.py 用同一条"严格早于当前行 time_ms"规则，训练用 BPR，
5 seed，与 BPR FM 的 0.5971 做严格对照。
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

# 主键 user，次键 author，第三键 time：按 (user, author) 分组，组内按时间排序。
order = np.lexsort((all_time, author_int, user_int))
u_sorted = user_int[order]; a_sorted = author_int[order]; t_sorted = all_time[order]
lv_sorted = np.asarray(all_lv)[order]
split_sorted = np.asarray(all_split)[order]; pos_sorted = np.asarray(all_pos)[order]

gap_ms = np.full(len(order), -1, dtype=np.int64)   # -1 = 该用户此前从未 long_view 过这个作者
last_lv_time = None
prev_key = None
for i in range(len(order)):
    key = (u_sorted[i], a_sorted[i])
    if key != prev_key:
        last_lv_time = None
        prev_key = key
    if last_lv_time is not None:
        gap_ms[i] = t_sorted[i] - last_lv_time
    if lv_sorted[i]:
        last_lv_time = t_sorted[i]

n_never = int(np.sum(gap_ms < 0))
total = len(order)
print(f"从未 long_view 过该作者的行数: {n_never} / {total} ({100*n_never/total:.2f}%)")

# 分桶：train 内有真实 gap 的行，按小时算分位数切 10 桶；"never" 单独占第 0 类。
split_of = np.asarray(all_split)
is_train = split_of[order] == split_names.index('train')
gap_hours_train = gap_ms[is_train & (gap_ms >= 0)] / 3_600_000.0
bucket_edges = np.quantile(gap_hours_train, np.linspace(0, 1, 11)[1:-1])   # 9 条边，10 桶

feat = {name: np.zeros(len(splits[name]), dtype=np.int32) for name in split_names}
for i in range(len(order)):
    si, p = split_sorted[i], pos_sorted[i]
    if gap_ms[i] < 0:
        feat[split_names[si]][p] = 0
    else:
        gh = gap_ms[i] / 3_600_000.0
        feat[split_names[si]][p] = 1 + int(np.searchsorted(bucket_edges, gh))   # 1..10

# 复用 data.py 的 5 域词表，追加第 6 域：author_recency ∈ {0..10}，11 类，不需要 UNK 槽。
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
dim = dim5 + 11

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
print(f"\n=== 6-field BPR FM (+author_recency), 5 seeds ===")
for name, vals in [('GAUC', gauc), ('nDCG@5', ndcg), ('primary', primary)]:
    print(f"  {name:8s} mean={statistics.mean(vals):.4f} std={statistics.pstdev(vals):.4f}")
print("  vs. BPR FM (5-field, RUN_LOG.md): GAUC 0.6638 | nDCG@5 0.5304 | primary 0.5971")
print("  vs. BPR FM + prior_exposure (RUN_LOG.md): GAUC 0.6662 | nDCG@5 0.5310 | primary 0.5986")
