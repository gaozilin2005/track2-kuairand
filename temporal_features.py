"""跨行时序特征：prior_exposure（是否精确复看过这个视频）+ author_recency
（离上次看这个作者的作品过了多久，分桶）。两个都已经在 RUN_LOG.md 里验证过有稳定
收益（ablation_prior_exposure.py +0.0015，ablation_author_recency.py +0.0017），
这里把它们从一次性诊断脚本收编成 encode() 默认会用的正式特征，供 baseline.py 和
sequence_model.py 共用。跟 sequence.py 用同一条"严格早于当前行 time_ms"规则，
numpy-only，不依赖 data.py（避免循环 import），行 tuple 布局按 data.load() 的约定
本地写死。
"""
import numpy as np

_DATE, _USER, _VIDEO, _AUTHOR, _TAB, _DUR, _LABEL, _TIME = range(8)

N_RECENCY_BUCKETS = 10   # + 1 个 "never" 类 = 11


def build_temporal_features(splits):
    """返回 {split: (N,2) int32}：col0=prior_exposure ∈ {0,1}，col1=author_recency ∈ {0..10}
    （0 = 该用户此前从未 long_view 过这个作者，1..10 = 按 train 内分位数切的时间桶，
    越小越近）。"""
    split_names = list(splits.keys())
    all_user, all_time, all_vid, all_author, all_lv, all_split, all_pos = [], [], [], [], [], [], []
    for si, name in enumerate(split_names):
        for pos, x in enumerate(splits[name]):
            all_user.append(x[_USER]); all_time.append(x[_TIME])
            all_vid.append(x[_VIDEO]); all_author.append(x[_AUTHOR])
            all_lv.append(x[_LABEL]); all_split.append(si); all_pos.append(pos)

    all_time = np.asarray(all_time, dtype=np.int64)
    all_lv = np.asarray(all_lv, dtype=bool)
    all_split = np.asarray(all_split, dtype=np.int32)
    all_pos = np.asarray(all_pos, dtype=np.int64)
    _, user_int = np.unique(all_user, return_inverse=True)
    _, author_int = np.unique(all_author, return_inverse=True)

    out = {name: np.zeros((len(splits[name]), 2), dtype=np.int32) for name in split_names}

    # --- prior_exposure：该用户此前 long_view 过这个确切视频吗 ---
    order = np.lexsort((all_time, user_int))
    seen = {}
    for idx in order:
        u = user_int[idx]; vid = all_vid[idx]
        s = seen.setdefault(u, set())
        if vid in s:
            out[split_names[all_split[idx]]][all_pos[idx], 0] = 1
        if all_lv[idx]:
            s.add(vid)

    # --- author_recency：距离上次 long_view 这个作者的作品过了多久（分桶） ---
    order2 = np.lexsort((all_time, author_int, user_int))   # 主键 user，次键 author，第三键 time
    u2 = user_int[order2]; a2 = author_int[order2]; t2 = all_time[order2]
    lv2 = all_lv[order2]; split2 = all_split[order2]; pos2 = all_pos[order2]

    gap_ms = np.full(len(order2), -1, dtype=np.int64)
    last_lv_time = None; prev_key = None
    for i in range(len(order2)):
        key = (u2[i], a2[i])
        if key != prev_key:
            last_lv_time = None; prev_key = key
        if last_lv_time is not None:
            gap_ms[i] = t2[i] - last_lv_time
        if lv2[i]:
            last_lv_time = t2[i]

    train_idx = split_names.index('train')
    is_train = split2 == train_idx
    gap_hours_train = gap_ms[is_train & (gap_ms >= 0)] / 3_600_000.0
    edges = np.quantile(gap_hours_train, np.linspace(0, 1, N_RECENCY_BUCKETS + 1)[1:-1])

    for i in range(len(order2)):
        if gap_ms[i] >= 0:
            gh = gap_ms[i] / 3_600_000.0
            out[split_names[split2[i]]][pos2[i], 1] = 1 + int(np.searchsorted(edges, gh))
        # gap_ms[i] < 0（"never"）保留初始化的 0，不需要特殊赋值

    return out
