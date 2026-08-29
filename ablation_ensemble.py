"""种子集成：把 N 个不同 seed 训出来的模型的**预测分**平均，而不是只平均它们的指标。

之前每个实验都跑 5 个 seed，但只用来算 mean±std（衡量稳定性）——从来没把这 5 个模型的
预测合起来用过。这是标准的方差削减手段，成本几乎为零（模型本来就要训），而且跟本项目
试过的所有方向都不一样：它不需要数据里有更多信号，只是把已有信号提取得更稳。

诚实的预期：集成削减的是**方差**，而本项目的天花板看起来是**偏差/信息**上限（九条独立
路线都收敛到 0.601~0.602）。所以收益应该有限。但因为几乎免费，值得实测。

注意打分口径：evaluate() 只用组内相对大小，但不同 seed 的分数尺度/偏移不一定可比，
直接平均原始分可能被某个尺度大的 seed 主导。所以同时测两种聚合方式：
  - raw：直接平均原始分数
  - rank：每个模型先在**用户组内**转成 rank（对单调变换不变，尺度无关），再平均
rank 平均在集成排序模型时通常更稳，这里一并对照。
"""
import argparse, collections, statistics
import numpy as np
from data import load, encode, watch_time_targets, FIELDS
from evaluate import evaluate
import baseline as B


def train_one(splits, enc, dim, wt, seed, epochs=40, bs=8192, patience=4,
              k=16, lr=0.001, aux_weight=0.2):
    """训一个跟当前默认配置完全一致的模型（7 域 + BPR + watchtime 辅助任务）。"""
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
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

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                       for pool, c in zip(neg_pools, counts)])
        perm = rng.permutation(len(pos_idx_all))
        pi, ni = pos_idx_all[perm], neg_idx_all[perm]
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
    return m


def groupwise_rank(users, scores):
    """把分数在每个用户组内转成 [0,1] 的 rank（尺度无关）。"""
    out = np.empty(len(scores), dtype=np.float64)
    by = collections.defaultdict(list)
    for i, u in enumerate(users):
        by[u].append(i)
    for u, idxs in by.items():
        idxs = np.array(idxs)
        s = scores[idxs]
        order = np.argsort(s)
        r = np.empty(len(s), dtype=np.float64)
        r[order] = np.arange(len(s))
        out[idxs] = r / max(len(s) - 1, 1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--n_models', type=int, default=5)
    a = ap.parse_args()

    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    enc, dim = encode(splits)
    wt = watch_time_targets(splits)
    Xte, yte, ute = enc['test']

    raw_scores = []
    print(f"\ntraining {a.n_models} models (seeds 0..{a.n_models-1}) ...")
    for seed in range(a.n_models):
        m = train_one(splits, enc, dim, wt, seed)
        s = m.predict(Xte)
        raw_scores.append(s)
        r = evaluate(ute, yte, s)
        print(f"  seed {seed}: GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")

    singles = [evaluate(ute, yte, s)['primary'] for s in raw_scores]
    print(f"\n  single-model mean primary: {statistics.mean(singles):.4f} ± {statistics.pstdev(singles):.4f}")

    # 逐步增加集成规模，看收益随模型数的变化（也用来判断是不是已经饱和）。
    ranked = [groupwise_rank(ute, s) for s in raw_scores]
    print("\n  ensemble size | raw-avg primary | rank-avg primary")
    for n in range(2, a.n_models + 1):
        r_raw = evaluate(ute, yte, np.mean(raw_scores[:n], axis=0))
        r_rank = evaluate(ute, yte, np.mean(ranked[:n], axis=0))
        print(f"     {n}          {r_raw['primary']:.4f}            {r_rank['primary']:.4f}")

    r_raw = evaluate(ute, yte, np.mean(raw_scores, axis=0))
    r_rank = evaluate(ute, yte, np.mean(ranked, axis=0))
    print(f"\n=== ensemble of {a.n_models} (raw avg) ===")
    print(f"  GAUC {r_raw['GAUC']:.4f} | nDCG@5 {r_raw['nDCG@5']:.4f} | primary {r_raw['primary']:.4f}")
    print(f"=== ensemble of {a.n_models} (groupwise-rank avg) ===")
    print(f"  GAUC {r_rank['GAUC']:.4f} | nDCG@5 {r_rank['nDCG@5']:.4f} | primary {r_rank['primary']:.4f}")
    print(f"\n  vs current best single-model config (RUN_LOG.md): primary 0.6017 ± 0.0004")
