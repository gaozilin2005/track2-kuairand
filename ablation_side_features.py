"""测试两个从没碰过的官方数据文件：`user_features_pure.csv`（用户画像）和
`video_features_statistic_pure.csv`（视频全站聚合统计）。

跟这个项目里其它"加特征"实验的本质区别：之前加的都是**从我们自己这段日志里
统计出来的时序特征**（prior_exposure/author_recency/click 历史），信息来源仍然是
同一批 140 万行交互。这两个文件是**官方随数据集一起发布的画像/统计文件**，
覆盖的是全平台范围（video_features_statistic 的计数明显比我们日志里能看到的量级
大得多——说明是全站统计，不是从我们这 140 万行里数出来的），是真正意义上"模型
从没见过的信息源"，不是同一口井里再打一遍水。

理论依据（不是"加特征就试试"）：本项目很早就验证过一个结论——**纯用户侧特征
（在组内恒定、不参与交互项的话）对组内排序贡献恒为零**（README「从哪里开始改」
一节有证明）。但 FM 的机制不是简单相加：只要新特征**作为独立域参与双线性交互项**
（拥有自己的 embedding，和 video_id/author_id 的 embedding 做内积），哪怕这个值
对同一用户的所有曝光恒定，它的 embedding 跟不同视频 embedding 的内积**依然会
随视频变化**——这正是 user_id 自己能影响排序的同一个机制。用户画像域相当于给
"user_id 的 embedding"配了一个可以在**同画像用户之间共享统计强度**的备胎：
不活跃用户自己的 user_id embedding 数据不够，但"注册 730+ 天、粉丝数 [1,10)"
这个桶里可能有几千个用户共享，能学得准得多。这正是 LightFM（Kula, 2015,
"Metadata Embeddings for Cold-start Recommendations"）解决冷启动的核心机制。

对照基准是当前最优配置（7 域 + BPR + watchtime 辅助任务）。
"""
import argparse, collections, csv, statistics
import numpy as np
from data import load, build_vocab, watch_time_targets, _raw_fields
from temporal_features import build_temporal_features
from evaluate import evaluate
import baseline as B

_USER, _VIDEO = 1, 2

USER_COLS = ['user_active_degree', 'follow_user_num_range', 'fans_user_num_range',
             'friend_user_num_range', 'register_days_range']
N_VIDEO_BUCKETS = 10


def load_user_features(path='./KuaiRand-Pure/data/user_features_pure.csv'):
    raw = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            raw[r['user_id']] = tuple(r[c] for c in USER_COLS)
    vocabs = [dict() for _ in USER_COLS]
    coded = {}
    for uid, vals in raw.items():
        codes = []
        for i, v in enumerate(vals):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
            codes.append(vocabs[i][v])
        coded[uid] = codes
    dims = [len(v) for v in vocabs]
    print(f'  user features: {len(coded)} users, field cardinalities {dict(zip(USER_COLS, dims))}')
    return coded, dims


def load_video_stat_features(path='./KuaiRand-Pure/data/video_features_statistic_pure.csv'):
    show_cnt, completion = {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            vid = r['video_id']
            show_cnt[vid] = float(r['show_cnt'])
            play = float(r['play_cnt'])
            completion[vid] = float(r['complete_play_cnt']) / play if play > 0 else 0.0
    show_edges = np.quantile(list(show_cnt.values()), np.linspace(0, 1, N_VIDEO_BUCKETS + 1)[1:-1])
    comp_edges = np.quantile(list(completion.values()), np.linspace(0, 1, N_VIDEO_BUCKETS + 1)[1:-1])
    show_code = {v: int(np.searchsorted(show_edges, s)) for v, s in show_cnt.items()}
    comp_code = {v: int(np.searchsorted(comp_edges, c)) for v, c in completion.items()}
    print(f'  video stat features: {len(show_cnt)} videos, 2 fields x {N_VIDEO_BUCKETS} buckets')
    return show_code, comp_code


def encode_with_side(splits, use_user, use_video):
    """7 个默认域 + 选中的画像/统计域。跟 ablation_other_signals.py 同一套构造方式。"""
    vocabs, unk, field_dims, offsets, edges = build_vocab(splits)
    temporal = build_temporal_features(splits)
    dim5 = int(sum(field_dims))
    base_dims = [2, 11]  # prior_exposure, author_recency
    extra_dims = []

    if use_user:
        user_coded, user_dims = load_user_features()
        extra_dims += [d + 1 for d in user_dims]  # +1 UNK 槽（理论上 100% 覆盖，UNK 不会被用到）
    if use_video:
        show_code, comp_code = load_video_stat_features()
        extra_dims += [N_VIDEO_BUCKETS + 1, N_VIDEO_BUCKETS + 1]

    all_dims = base_dims + extra_dims
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
            y[n] = x[6]
            users.append(x[_USER])
        X[:, 5] = temporal[name][:, 0] + offs[0]
        X[:, 6] = temporal[name][:, 1] + offs[1]

        col = 7
        if use_user:
            unk_codes = user_dims  # 每个域的 UNK 槽就是该域的基数（vocab 之后紧跟的下一个整数）
            for i in range(len(USER_COLS)):
                vals = np.array([user_coded.get(x[_USER], unk_codes)[i] for x in rws], dtype=np.int32)
                X[:, col] = vals + offs[2 + col - 7]
                col += 1
        if use_video:
            unk_show, unk_comp = N_VIDEO_BUCKETS, N_VIDEO_BUCKETS
            X[:, col] = np.array([show_code.get(x[_VIDEO], unk_show) for x in rws], dtype=np.int32) + offs[2 + col - 7]
            col += 1
            X[:, col] = np.array([comp_code.get(x[_VIDEO], unk_comp) for x in rws], dtype=np.int32) + offs[2 + col - 7]
            col += 1
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
    wt = watch_time_targets(splits)

    configs = [(False, False, '当前默认 7 域（对照）'),
               (True, False, '+ 用户画像（5 域）'),
               (False, True, '+ 视频全站统计（2 域）'),
               (True, True, '+ 用户画像 + 视频统计（7 域，共 14 域）')]

    for use_u, use_v, desc in configs:
        enc, dim = encode_with_side(splits, use_u, use_v)
        res = [run(enc, dim, wt, s) for s in range(a.seeds)]
        pri = [r['primary'] for r in res]
        g = [r['GAUC'] for r in res]; nd = [r['nDCG@5'] for r in res]
        print(f"\n=== {desc} ({enc['train'][0].shape[1]} 域) ===")
        print(f"  GAUC    {statistics.mean(g):.4f} ± {statistics.pstdev(g):.4f}")
        print(f"  nDCG@5  {statistics.mean(nd):.4f} ± {statistics.pstdev(nd):.4f}")
        print(f"  primary {statistics.mean(pri):.4f} ± {statistics.pstdev(pri):.4f}")
        print(f"  per-seed: {', '.join(f'{p:.4f}' for p in pri)}")
