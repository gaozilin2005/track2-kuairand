"""便宜的诊断实验：只加一个 0/1 特征——"这个用户之前 long_view 过这个确切视频吗"——
不建 attention，看任何序列信号对这个数据集有没有用。几分钟内出结果，用来决定
要不要投入 BST/DIEN 这类更重的序列架构（见 RUN_LOG.md 的 DIN 实验）。

跟 sequence.py 用同一条"严格早于当前行 time_ms"规则（跨 train/valid/test 全量日志
时间排序），但这里只留一个 bit，不留完整历史窗口——如果连这个都没用，attention
更不太可能有用；如果这个有用，才值得投入更重的架构。
训练用 BPR（与当前最优配置一致），5 seed，与 BPR FM 的 0.5971 做严格对照。
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

all_user, all_time, all_vid, all_lv, all_split, all_pos = [], [], [], [], [], []
for si, name in enumerate(split_names):
    for pos, x in enumerate(splits[name]):
        all_user.append(x[_USER]); all_time.append(x[_TIME])
        all_vid.append(x[_VIDEO]); all_lv.append(x[_LABEL])
        all_split.append(si); all_pos.append(pos)

all_time = np.asarray(all_time, dtype=np.int64)
_, user_int = np.unique(all_user, return_inverse=True)
order = np.lexsort((all_time, user_int))   # 主键 user，次键 time

feat = {name: np.zeros(len(splits[name]), dtype=np.int32) for name in split_names}
seen = {}   # user_int -> 该用户此前 long_view 过的 video_id 集合（随时间扫描增长）
n_hit = 0
for idx in order:
    u = user_int[idx]
    vid = all_vid[idx]
    s = seen.setdefault(u, set())
    if vid in s:
        feat[split_names[all_split[idx]]][all_pos[idx]] = 1
        n_hit += 1
    if all_lv[idx]:
        s.add(vid)
total = len(order)
print(f"prior_exposure=1 的行数: {n_hit} / {total} ({100*n_hit/total:.2f}%)")

# 复用 data.py 的 5 域词表，追加第 6 域：prior_exposure ∈ {0,1}，不需要 UNK 槽。
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
print(f"\n=== 6-field BPR FM (+prior_exposure), 5 seeds ===")
for name, vals in [('GAUC', gauc), ('nDCG@5', ndcg), ('primary', primary)]:
    print(f"  {name:8s} mean={statistics.mean(vals):.4f} std={statistics.pstdev(vals):.4f}")
print("  vs. BPR FM (5-field, RUN_LOG.md): GAUC 0.6638 | nDCG@5 0.5304 | primary 0.5971")
