"""第三个"CWM 方向"诊断：把观看时长当**特征**用，不是辅助 loss——跟
prior_exposure/author_recency 同一套打法（编码用户过去跟某作者的互动），但这次编码的
是"参与深度"（历史观看比例的均值），而不是"有没有见过"（prior_exposure）或"多久之前"
（author_recency），测试这个维度是不是新增量，还是已经被前两个时序特征间接覆盖了。

跟 ablation_author_recency.py 同一条"严格早于当前行 time_ms"规则，同一套 (user,author)
分组扫描。极端循环播放（>10x duration，噪声，不是参与度信号，跟 watch_time_targets 里
的删失回归诊断用的是同一个发现）在算历史均值前先截断，避免污染。

单独作为第 6 域接到原始 5 域 FM 上，BPR，5 seed，与 BPR FM 的 0.5971 做严格对照
（不是接到当前 7 域 + watchtime 默认之上——这是测这个信号本身有没有用，不是测它
在已经很强的配置上还能不能挤出增量）。
"""
import statistics
import numpy as np
from data import load, build_vocab
from evaluate import evaluate
import baseline as B
import collections

D = './KuaiRand-Pure/data'
splits = load(D)
print({k: len(v) for k, v in splits.items()})

_DATE, _USER, _VIDEO, _AUTHOR, _TAB, _DUR, _LABEL, _TIME, _CLICK, _PT = range(10)
split_names = list(splits.keys())

all_user, all_time, all_author, all_split, all_pos, all_ratio = [], [], [], [], [], []
for si, name in enumerate(split_names):
    for pos, x in enumerate(splits[name]):
        all_user.append(x[_USER]); all_time.append(x[_TIME])
        all_author.append(x[_AUTHOR]); all_split.append(si); all_pos.append(pos)
        dur = max(x[_DUR], 1.0)
        pt_capped = min(x[_PT], dur * 10)   # 截断极端循环播放（同 watch_time_targets 的发现）
        all_ratio.append(np.log1p(pt_capped) / 12.0)

all_time = np.asarray(all_time, dtype=np.int64)
all_ratio = np.asarray(all_ratio, dtype=np.float64)
all_split = np.asarray(all_split); all_pos = np.asarray(all_pos)
_, user_int = np.unique(all_user, return_inverse=True)
_, author_int = np.unique(all_author, return_inverse=True)

order = np.lexsort((all_time, author_int, user_int))   # 主键 user，次键 author，第三键 time
u_sorted = user_int[order]; a_sorted = author_int[order]
ratio_sorted = all_ratio[order]
split_sorted = all_split[order]; pos_sorted = all_pos[order]

feat = {name: np.zeros(len(splits[name]), dtype=np.float64) for name in split_names}
running_sum = {}; running_count = {}
for i in range(len(order)):
    key = (u_sorted[i], a_sorted[i])
    c = running_count.get(key, 0)
    if c > 0:
        feat[split_names[split_sorted[i]]][pos_sorted[i]] = running_sum[key] / c
    running_sum[key] = running_sum.get(key, 0.0) + ratio_sorted[i]
    running_count[key] = c + 1

n_have_prior = sum(int((feat[n] > 0).sum()) for n in split_names)
total = len(order)
print(f"有历史参与度可查的行数: {n_have_prior} / {total} ({100*n_have_prior/total:.2f}%)")

# 分桶：train 内有历史值的行，按分位数切 10 桶；"never"（值为 0）单独占第 0 类。
train_vals = feat['train'][feat['train'] > 0]
edges = np.quantile(train_vals, np.linspace(0, 1, 11)[1:-1])

bucketed = {}
for name in split_names:
    v = feat[name]
    b = np.zeros(len(v), dtype=np.int32)
    mask = v > 0
    b[mask] = 1 + np.searchsorted(edges, v[mask])
    bucketed[name] = b

# 复用 data.py 的 5 域词表，追加第 6 域：author_watch_affinity ∈ {0..10}，11 类。
vocabs, unk, field_dims, offsets, edges2 = build_vocab(splits)
dim5 = int(sum(field_dims))
_raw = lambda x: [x[_USER], x[_VIDEO], x[_AUTHOR], x[_TAB], str(int(np.searchsorted(edges2, x[_DUR])))]

enc = {}
for name, rws in splits.items():
    X = np.empty((len(rws), 6), dtype=np.int32)
    y = np.empty(len(rws), dtype=np.float32)
    users = []
    for n, x in enumerate(rws):
        for i, v in enumerate(_raw(x)):
            X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
        X[n, 5] = dim5 + bucketed[name][n]
        y[n] = x[_LABEL]
        users.append(x[_USER])
    enc[name] = (X, y, users)
dim = dim5 + 11

Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

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
print(f"\n=== 6-field BPR FM (+author_watch_affinity), 5 seeds ===")
for name, vals in [('GAUC', gauc), ('nDCG@5', ndcg), ('primary', primary)]:
    print(f"  {name:8s} mean={statistics.mean(vals):.4f} std={statistics.pstdev(vals):.4f}")
print("  vs. BPR FM (5-field, RUN_LOG.md): GAUC 0.6638 | nDCG@5 0.5304 | primary 0.5971")
print("  vs. BPR FM + author_recency:      GAUC 0.6663 | nDCG@5 0.5313 | primary 0.5988")
