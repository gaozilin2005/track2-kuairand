"""用户历史序列构建（numpy-only，不依赖 torch）。

为每一行构造该用户在此刻之前（严格早于该行 time_ms）最近 L 个 long_view=1
video_id 的定长历史（左 padding，最近的一条落在最后一个槽位）。
历史跨越 train/valid/test 全部日志构建（晚期的行可以看到早期任意 split 里
的历史事件），但只在各自 split 内输出，行序与 encode(splits)[split] 完全对齐。
"""
import numpy as np

# splits 里每行 tuple 的字段位置（与 data.load() 保持一致）
_DATE, _USER, _VIDEO, _AUTHOR, _TAB, _DUR, _LABEL, _TIME = range(8)

def build_history(splits, vocabs, unk, offsets, pad_idx, L=160):
    """返回 {split_name: (N_split, L) int32 array}，pad 值为 pad_idx。"""
    video_vocab, video_unk, video_offset = vocabs[1], unk[1], offsets[1]
    split_names = list(splits.keys())

    # 拼接全量行，同时记录每行属于哪个 split 的哪个位置（顺序与 encode() 一致）。
    all_user, all_time, all_vid, all_lv, all_split, all_pos = [], [], [], [], [], []
    for si, name in enumerate(split_names):
        rws = splits[name]
        for pos, x in enumerate(rws):
            all_user.append(x[_USER])
            all_time.append(x[_TIME])
            all_vid.append(video_vocab.get(x[_VIDEO], video_unk) + video_offset)
            all_lv.append(x[_LABEL])
            all_split.append(si)
            all_pos.append(pos)

    all_time = np.asarray(all_time, dtype=np.int64)
    all_vid = np.asarray(all_vid, dtype=np.int32)
    all_lv = np.asarray(all_lv, dtype=bool)
    all_split = np.asarray(all_split, dtype=np.int8)
    all_pos = np.asarray(all_pos, dtype=np.int64)
    _, user_int = np.unique(all_user, return_inverse=True)

    hist = {name: np.full((len(splits[name]), L), pad_idx, dtype=np.int32)
            for name in split_names}

    order = np.lexsort((all_time, user_int))          # 主键 user，次键 time（组内按时间排序）
    u_sorted = user_int[order]
    time_sorted = all_time[order]
    vid_sorted = all_vid[order]
    lv_sorted = all_lv[order]
    split_sorted = all_split[order]
    pos_sorted = all_pos[order]

    boundaries = np.flatnonzero(np.diff(u_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(u_sorted)]))

    n_zero_history_rows = 0
    for start, end in zip(starts, ends):
        seg_len = end - start
        seg_lv = lv_sorted[start:end]
        seg_vid = vid_sorted[start:end]
        seg_split = split_sorted[start:end]
        seg_pos = pos_sorted[start:end]

        ev_positions = np.flatnonzero(seg_lv)          # 该用户 long_view=1 事件的局部位置（已按时间排序）
        n_ev = len(ev_positions)

        if n_ev == 0:
            block = np.full((seg_len, L), pad_idx, dtype=np.int32)
            n_zero_history_rows += seg_len
        else:
            ev_vids = seg_vid[ev_positions]
            # num_prior[p] = 该用户在位置 p 之前（不含 p 本身）发生的 long_view 事件数
            num_prior = np.searchsorted(ev_positions, np.arange(seg_len), side='left')
            idxs = num_prior[:, None] - L + np.arange(L)[None, :]     # (seg_len, L)
            valid = idxs >= 0
            gathered = ev_vids[np.clip(idxs, 0, n_ev - 1)]
            block = np.where(valid, gathered, pad_idx).astype(np.int32)
            n_zero_history_rows += int(np.sum(num_prior == 0))

        for si, name in enumerate(split_names):
            m = seg_split == si
            if np.any(m):
                hist[name][seg_pos[m]] = block[m]

    total_rows = sum(len(splits[n]) for n in split_names)
    print(f"  history: L={L}, {total_rows} rows, "
          f"{n_zero_history_rows} ({100*n_zero_history_rows/total_rows:.1f}%) with zero prior long_view history")
    return hist
