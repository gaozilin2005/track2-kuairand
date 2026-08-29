"""Tier 2：把"历史特征"这套已经证明有效的做法，套用到 long_view 以外的反馈信号上。

依据：`prior_exposure`（+0.0015）和 `author_recency`（+0.0017）是本项目最有效的两个特征，
两个都是**跨行历史**特征，都建在 `long_view` 上。同样的构造套到 `is_click` / 强互动信号上
从来没试过。

注意跟已否决实验的区别：`is_click` 作为**辅助任务**是空结果（0.6007，两个权重都试过）——
那测的是"预测点击能不能让 embedding 变好"。这里测的是完全不同的东西：**"这个用户以前
点过这个视频/这个作者吗"**，是关于这一对具体 (user, item) 的时序信号，跟 prior_exposure
的机制一样，不是同一个实验。

三个候选特征（都用跟 temporal_features.py 相同的"严格早于当前行 time_ms"规则）：
  prior_click          该用户此前点击过这个确切视频吗（is_click 版的 prior_exposure，
                       is_click 密度 46% 远高于 long_view 的 34%，命中率应该更高）
  author_click_recency 离上次点击这个作者的作品过了多久（分桶，author_recency 的点击版）
  author_engage        该用户此前对这个作者做过强互动吗（like/follow/comment/forward/
                       profile_enter 合并，单个都太稀疏——最密的 profile_enter 也才 2.5%，
                       合并后 4.5%；prior_exposure 只有 0.2% 命中率也拿到了 +0.0015，
                       所以稀疏本身不是问题，关键看命中时信号强不强）

对照基准是**当前最优配置**（7 域 + BPR + watchtime 辅助任务，0.6017/单 seed 0.6020），
不是 5 域基线——这里问的是"在现有最好的基础上还能不能加东西"。
"""
import argparse, collections, statistics
import numpy as np
from data import (load, build_vocab, watch_time_targets, _raw_fields, FIELDS)
from evaluate import evaluate
import baseline as B

_DATE, _USER, _VIDEO, _AUTHOR, _TAB, _DUR, _LABEL, _TIME, _CLICK, _PT, _ENGAGE = range(11)

N_BUCKETS = 10


def build_extra_features(splits):
    """返回 {split: (N,3) int32}：prior_click / author_click_recency / author_engage。"""
    split_names = list(splits.keys())
    cols = {'user': [], 'time': [], 'vid': [], 'author': [],
            'click': [], 'engage': [], 'split': [], 'pos': []}
    for si, name in enumerate(split_names):
        for pos, x in enumerate(splits[name]):
            cols['user'].append(x[_USER]); cols['time'].append(x[_TIME])
            cols['vid'].append(x[_VIDEO]); cols['author'].append(x[_AUTHOR])
            cols['click'].append(x[_CLICK]); cols['engage'].append(x[_ENGAGE])
            cols['split'].append(si); cols['pos'].append(pos)

    time = np.asarray(cols['time'], dtype=np.int64)
    click = np.asarray(cols['click'], dtype=bool)
    engage = np.asarray(cols['engage'], dtype=bool)
    split_arr = np.asarray(cols['split'], dtype=np.int32)
    pos_arr = np.asarray(cols['pos'], dtype=np.int64)
    _, user_int = np.unique(cols['user'], return_inverse=True)
    _, author_int = np.unique(cols['author'], return_inverse=True)

    out = {name: np.zeros((len(splits[name]), 3), dtype=np.int32) for name in split_names}

    # --- 1) prior_click：该用户此前 is_click 过这个确切 video ---
    order = np.lexsort((time, user_int))
    seen = {}
    for idx in order:
        u, v = user_int[idx], cols['vid'][idx]
        s = seen.setdefault(u, set())
        if v in s:
            out[split_names[split_arr[idx]]][pos_arr[idx], 0] = 1
        if click[idx]:
            s.add(v)

    # --- 2/3) 按 (user, author) 分组扫描：点击时间间隔 + 强互动是否发生过 ---
    order2 = np.lexsort((time, author_int, user_int))
    u2, a2, t2 = user_int[order2], author_int[order2], time[order2]
    click2, eng2 = click[order2], engage[order2]
    sp2, ps2 = split_arr[order2], pos_arr[order2]

    gap = np.full(len(order2), -1, dtype=np.int64)
    had_engage = np.zeros(len(order2), dtype=bool)
    last_click_t, seen_engage, prev_key = None, False, None
    for i in range(len(order2)):
        key = (u2[i], a2[i])
        if key != prev_key:
            last_click_t, seen_engage, prev_key = None, False, key
        if last_click_t is not None:
            gap[i] = t2[i] - last_click_t
        had_engage[i] = seen_engage
        if click2[i]:
            last_click_t = t2[i]
        if eng2[i]:
            seen_engage = True

    train_idx = split_names.index('train')
    is_tr = sp2 == train_idx
    gh_tr = gap[is_tr & (gap >= 0)] / 3_600_000.0
    edges = (np.quantile(gh_tr, np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
             if len(gh_tr) else np.zeros(N_BUCKETS - 1))

    for i in range(len(order2)):
        tgt = out[split_names[sp2[i]]]
        if gap[i] >= 0:
            tgt[ps2[i], 1] = 1 + int(np.searchsorted(edges, gap[i] / 3_600_000.0))
        tgt[ps2[i], 2] = 1 if had_engage[i] else 0

    tot = len(order2)
    print(f"  prior_click fires on          {sum(int((out[n][:,0]==1).sum()) for n in split_names):7d} / {tot} rows")
    print(f"  author_click_recency non-zero {sum(int((out[n][:,1]>0).sum()) for n in split_names):7d} / {tot} rows")
    print(f"  author_engage fires on        {sum(int((out[n][:,2]==1).sum()) for n in split_names):7d} / {tot} rows")
    return out


EXTRA_DIMS = {'prior_click': 2, 'author_click_recency': N_BUCKETS + 1, 'author_engage': 2}
EXTRA_COL = {'prior_click': 0, 'author_click_recency': 1, 'author_engage': 2}


def encode_with_extras(splits, extras, use):
    """7 个默认域 + 选中的 extra 域。use 是 EXTRA_COL 的键列表。"""
    from temporal_features import build_temporal_features
    vocabs, unk, field_dims, offsets, edges = build_vocab(splits)
    temporal = build_temporal_features(splits)
    dim5 = int(sum(field_dims))
    base_dims = [2, N_BUCKETS + 1]                      # prior_exposure, author_recency
    all_dims = base_dims + [EXTRA_DIMS[u] for u in use]
    offs = np.cumsum([dim5] + all_dims[:-1]).astype(np.int64)
    dim = dim5 + sum(all_dims)
    n_fields = 5 + len(all_dims)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), n_fields), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(_raw_fields(x, edges)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[_LABEL]
            users.append(x[_USER])
        X[:, 5] = temporal[name][:, 0] + offs[0]
        X[:, 6] = temporal[name][:, 1] + offs[1]
        for j, u in enumerate(use):
            X[:, 7 + j] = extras[name][:, EXTRA_COL[u]] + offs[2 + j]
        enc[name] = (X, y, users)
    return enc, int(dim)


def run(enc, dim, wt, seed, epochs=40, bs=8192, patience=4, k=16, lr=0.001, aux_weight=0.2):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    t_arr, tau_arr, cens_arr = wt['train']
    m = B.FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
    for i, (u, yv) in enumerate(zip(utr, ytr)):
        (user_pos if yv > 0 else user_neg)[u].append(i)
    mixed = [u for u in user_pos if u in user_neg]
    pos_blocks = [np.array(user_pos[u]) for u in mixed]
    neg_pools = [np.array(user_neg[u]) for u in mixed]
    pos_idx_all = np.concatenate(pos_blocks)
    counts = np.array([len(b) for b in pos_blocks])

    best, state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        neg_idx = np.concatenate([rng.choice(p, size=c, replace=True)
                                   for p, c in zip(neg_pools, counts)])
        perm = rng.permutation(len(pos_idx_all))
        pi, ni = pos_idx_all[perm], neg_idx[perm]
        for i in range(0, len(pi), bs):
            bpi, bni = pi[i:i + bs], ni[i:i + bs]
            m.step_pairwise_watchtime(
                Xtr[bpi], Xtr[bni],
                t_arr[bpi], tau_arr[bpi], cens_arr[bpi],
                t_arr[bni], tau_arr[bni], cens_arr[bni], aux_weight=aux_weight)
        va = evaluate(uva, yva, m.predict(Xva))['primary']
        if va > best + 1e-5:
            best, bad, state = va, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = state
    return evaluate(ute, yte, m.predict(Xte))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()

    splits = load(a.data_dir)
    print({k: len(v) for k, v in splits.items()})
    print('\nbuilding extra history features ...')
    extras = build_extra_features(splits)
    wt = watch_time_targets(splits)

    configs = [([], '当前默认 7 域（对照）'),
               (['prior_click'], '+ prior_click'),
               (['author_click_recency'], '+ author_click_recency'),
               (['author_engage'], '+ author_engage'),
               (['prior_click', 'author_click_recency', 'author_engage'], '+ 三个全加')]

    for use, desc in configs:
        enc, dim = encode_with_extras(splits, extras, use)
        res = [run(enc, dim, wt, s) for s in range(a.seeds)]
        pri = [r['primary'] for r in res]
        g = [r['GAUC'] for r in res]; nd = [r['nDCG@5'] for r in res]
        print(f"\n=== {desc} ({enc['train'][0].shape[1]} 域) ===")
        print(f"  GAUC    {statistics.mean(g):.4f} ± {statistics.pstdev(g):.4f}")
        print(f"  nDCG@5  {statistics.mean(nd):.4f} ± {statistics.pstdev(nd):.4f}")
        print(f"  primary {statistics.mean(pri):.4f} ± {statistics.pstdev(pri):.4f}")
        print(f"  per-seed: {', '.join(f'{p:.4f}' for p in pri)}")
