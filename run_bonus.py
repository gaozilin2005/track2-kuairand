"""Bonus benchmark 跑法：KuaiRand-1K / 27K，**跟 Pure 完全同一套任务和指标**
（用户内排序，正例 long_view，GAUC / nDCG@5，primary = 两者平均，用同一个 evaluate.py）。

赛题原文确认三个变体口径一致："KuaiRand-Pure / KuaiRand-1k / KuaiRand-27k → GAUC / nDCG@5"。
所以这里不需要新指标，只需要能把更大的数据喂进同一套流程——瓶颈是内存（本机 8GB），
用 `data_large.py` 的列式加载器代替 `data.py` 的 list-of-tuples。

模型用 5 域 FM + BPR（Pure 上验证过 BPR > pointwise，+0.0025）。时序特征
（prior_exposure / author_recency）暂不带——那两个要做全量 (user, author) 时间扫描，
在这个数据量下是另一个量级的工程，先把 baseline 打通，有余力再加。

用法：
  python3 run_bonus.py --suffix 1k  --data_dir ./KuaiRand-1K/data
  python3 run_bonus.py --suffix 27k --data_dir ./KuaiRand-27K/data
"""
import argparse, collections, resource, time
import numpy as np

from data_large import load_columnar, encode_columnar
from evaluate import evaluate
import baseline as B


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def run_fm_bpr(enc, dim, k=16, lr=0.001, epochs=40, bs=8192, patience=4,
               seed=0, verbose=True):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = B.FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)

    if verbose:
        print(f'  building BPR pools ... (peak mem {peak_gb():.2f} GB)')
    user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
    for i, (u, yv) in enumerate(zip(utr, ytr)):
        (user_pos if yv > 0 else user_neg)[u].append(i)
    mixed = [u for u in user_pos if u in user_neg]
    pos_blocks = [np.array(user_pos[u], dtype=np.int64) for u in mixed]
    neg_pools = [np.array(user_neg[u], dtype=np.int64) for u in mixed]
    user_pos.clear(); user_neg.clear()
    pos_idx_all = np.concatenate(pos_blocks)
    counts = np.array([len(b) for b in pos_blocks])
    del pos_blocks
    if verbose:
        print(f'  {len(mixed)} mixed users, {len(pos_idx_all)} pos-anchored pairs/epoch '
              f'(peak mem {peak_gb():.2f} GB)')

    best, state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        neg_idx = np.concatenate([rng.choice(p, size=c, replace=True)
                                   for p, c in zip(neg_pools, counts)])
        perm = rng.permutation(len(pos_idx_all))
        pi, ni = pos_idx_all[perm], neg_idx[perm]
        for i in range(0, len(pi), bs):
            m.step_pairwise(Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]])
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, state = va['primary'], 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f'  early stop at epoch {ep}')
                break
    m.V, m.W, m.b = state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test': evaluate(ute, yte, m.predict(Xte)),
            'valid_best': best}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='1k', choices=['1k', '27k'])
    ap.add_argument('--data_dir', default='./KuaiRand-1K/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    t_start = time.time()
    print(f'loading {a.data_dir} (suffix={a.suffix}) ...')
    data, vocab = load_columnar(a.data_dir, suffix=a.suffix)
    print(f'  peak mem after load: {peak_gb():.2f} GB')
    enc, dim = encode_columnar(data, vocab)
    data.clear()
    print(f'  peak mem after encode: {peak_gb():.2f} GB')

    res = run_fm_bpr(enc, dim, k=a.k, epochs=a.epochs, seed=a.seed)
    print(f"\n=== KuaiRand-{a.suffix.upper()} : FM + BPR (5 fields), seed={a.seed} ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    print(f"  valid-best primary during training: {res['valid_best']:.4f}")
    print(f"  wall-clock {time.time()-t_start:.0f}s | peak mem {peak_gb():.2f} GB")
