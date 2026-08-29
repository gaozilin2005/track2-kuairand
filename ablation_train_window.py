"""诊断：训练集里 Apr 9-12 这 4 天占了全部训练数据的 64%（72.5 万行），
但它的曝光强度是 7.4 impressions/user/day，而 valid/test 只有约 1.1——
评测发生在低强度区间，训练数据的大头却来自一个结构上不一样的高强度区间
（用户群体是同一批，91% 的 test 用户在 early train 里出现过，所以不是人群变化，
是记录强度变化）。

这个脚本测：把高强度早期数据整段丢掉，只用后期（跟评测同分布）的数据训练，
会不会反而更好。丢掉 64% 的数据换来"每一行都跟评测分布一致"——如果分数不降甚至
上升，说明早期数据确实在拖后腿，那么下一步就该做加权（软版本）而不是硬截断。

跟其它 ablation_*.py 同一套口径：BPR + 当前默认的 7 域特征 + watchtime 辅助任务
（即完整的当前最优配置，0.6017），只改训练集的日期范围，5 seed。
"""
import argparse, collections, statistics
import numpy as np
from data import load, encode, watch_time_targets, FIELDS
from evaluate import evaluate
import baseline as B

_DATE = 0

def run_with_cutoff(splits, cutoff, seed, epochs=40, bs=8192, patience=4,
                    k=16, lr=0.001, aux_weight=0.2, half_life_days=None):
    """cutoff=None 表示用全部训练数据；否则只保留 date >= cutoff 的训练行。
    half_life_days 不为 None 时改用**软加权**（保留全部数据，但按距离训练集末尾的
    天数做指数衰减采样权重，半衰期 half_life_days 天）——硬截断的连续版本。
    valid/test 永远不动。"""
    enc, dim = encode(splits)
    wt = watch_time_targets(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    t_arr, tau_arr, cens_arr = wt['train']
    dates = np.array([x[_DATE] for x in splits['train']])

    if cutoff is not None:
        keep = dates >= cutoff
        Xtr = Xtr[keep]; ytr = ytr[keep]
        utr = [u for u, kp in zip(utr, keep) if kp]
        t_arr = t_arr[keep]; tau_arr = tau_arr[keep]; cens_arr = cens_arr[keep]
        dates = dates[keep]

    # 软加权：把 YYYYMMDD 转成"距离训练集最后一天多少天"，再算指数衰减权重。
    row_w = None
    if half_life_days is not None:
        import datetime
        def to_ord(d):
            s = str(d)
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:])).toordinal()
        ords = np.array([to_ord(d) for d in dates])
        age = ords.max() - ords
        row_w = 0.5 ** (age / half_life_days)

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
    # 软加权时，每个 epoch 按权重重采样正例锚点（保持每轮 pair 数不变，只改分布）。
    pos_p = None
    if row_w is not None:
        pw = row_w[pos_idx_all]
        pos_p = pw / pw.sum()

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                       for pool, c in zip(neg_pools, counts)])
        if pos_p is None:
            perm = rng.permutation(len(pos_idx_all))
            pi, ni = pos_idx_all[perm], neg_idx_all[perm]
        else:
            sel = rng.choice(len(pos_idx_all), size=len(pos_idx_all), replace=True, p=pos_p)
            pi, ni = pos_idx_all[sel], neg_idx_all[sel]
        for i in range(0, len(pi), bs):
            bpi, bni = pi[i:i + bs], ni[i:i + bs]
            m.step_pairwise_watchtime(
                Xtr[bpi], Xtr[bni],
                t_arr[bpi], tau_arr[bpi], cens_arr[bpi],
                t_arr[bni], tau_arr[bni], cens_arr[bni],
                aux_weight=aux_weight)
        va = evaluate(uva, yva, m.predict(Xva))['primary']
        if va > best + 1e-5:
            best, bad = va, 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return evaluate(ute, yte, m.predict(Xte)), len(ytr), len(mixed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    # (cutoff, half_life_days, 描述)
    # None/None = 全量（当前默认，0.6017 对照）；硬截断已实测越砍越差（见 RUN_LOG.md），
    # 这里主测软加权：保留全部数据，只是让靠近评测期的行采样更频繁。
    configs = [(None, None, '全部训练数据，均匀（当前默认）'),
               (None, 3.0,  '软加权：半衰期 3 天'),
               (None, 7.0,  '软加权：半衰期 7 天'),
               (None, 14.0, '软加权：半衰期 14 天（很温和）')]

    for cutoff, half_life, desc in configs:
        results, n_rows, n_users = [], None, None
        for seed in range(a.seeds):
            r, n_rows, n_users = run_with_cutoff(splits, cutoff, seed, half_life_days=half_life)
            results.append(r)
        pri = [r['primary'] for r in results]
        g = [r['GAUC'] for r in results]; nd = [r['nDCG@5'] for r in results]
        print(f"\n=== {desc} ===")
        print(f"  train rows={n_rows} ({100*n_rows/len(splits['train']):.1f}% of full), mixed users={n_users}")
        print(f"  GAUC    {statistics.mean(g):.4f} ± {statistics.pstdev(g):.4f}")
        print(f"  nDCG@5  {statistics.mean(nd):.4f} ± {statistics.pstdev(nd):.4f}")
        print(f"  primary {statistics.mean(pri):.4f} ± {statistics.pstdev(pri):.4f}")
        print(f"  per-seed: {', '.join(f'{p:.4f}' for p in pri)}")
