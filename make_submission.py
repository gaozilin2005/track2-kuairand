"""生成最终提交文件（KuaiRand-Pure，官方任务：用户内排序，long_view，GAUC/nDCG@5）。

⚠️ 为什么不用 `submit.py --make`：那段代码写死了 `m.step`（pointwise loss）+ 当时的 5 域
`encode()`。现在 `data.encode()` 默认返回 7 域（加了 prior_exposure / author_recency），
所以 `--make` 既不是官方 baseline、也不是我们的最优配置，只能当"生成一个格式合法的示例"用。
这个脚本按**验证集最优**的配置生成提交——赛题明确说 "The submission scored for ranking is
the validation-best checkpoint"。

两种模式：
  --mode single   7 域 FM + BPR + watchtime 辅助任务（valid primary 0.6070，可完全复现，
                  单核 CPU 几十秒）
  --mode ensemble 异构集成，读 scores/*.npz 里缓存的各成员分数，按 valid 选出的融合方式
                  加权（valid primary 0.6091）。需要先跑 ablation_hetero_ensemble.py
                  --member <name> 把成员分数缓存好。

模型选择、融合方式、成员子集**全部只在 valid 上决定**，test 只用来最后写分数，
不参与任何选择——避免在被评测的那个数字上调参。
"""
import argparse, collections, itertools, os
import numpy as np

from data import load, encode, watch_time_targets
from evaluate import evaluate
from submit import write_submission, read_submission
import baseline as B


def train_best_single(splits, seed=0, k=16, lr=0.001, epochs=40, bs=8192,
                      patience=4, aux_weight=0.2, verbose=True):
    """当前验证集最优的单模型配置：7 域 + BPR 主任务 + 观看时长删失回归辅助任务。
    早停完全按 valid primary，返回的是 valid-best 的那份参数（不是最后一轮）。"""
    enc, dim = encode(splits)
    wt = watch_time_targets(splits)
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
        p = evaluate(uva, yva, m.predict(Xva))['primary']
        if verbose:
            print(f"  epoch {ep:2d} | valid primary {p:.4f}")
        if p > best + 1e-5:
            best, bad, state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = state
    return m, enc, best


def _group_index(users):
    by = collections.defaultdict(list)
    for i, u in enumerate(users):
        by[u].append(i)
    return [np.array(v) for v in by.values()]


def _zscore(scores, groups):
    out = np.empty(len(scores))
    for idxs in groups:
        s = scores[idxs]
        sd = s.std()
        out[idxs] = (s - s.mean()) / (sd if sd > 1e-9 else 1.0)
    return out


def ensemble_scores(splits, scores_dir, target_split, verbose=True):
    """读缓存的成员分数，在 valid 上选最佳子集（z-score 融合），返回 target_split 的融合分。"""
    enc, _ = encode(splits)
    _, yva, uva = enc['valid']
    members = {}
    for fn in sorted(os.listdir(scores_dir)):
        if fn.endswith('.npz'):
            d = np.load(os.path.join(scores_dir, fn))
            members[fn[:-4]] = (d['valid'], d['test'])
    if not members:
        raise SystemExit(f'{scores_dir}/ 里没有成员分数，先跑 ablation_hetero_ensemble.py --member <name>')
    names = list(members)
    if verbose:
        print(f'  members: {names}')

    gva = _group_index(uva)
    tva = {n: _zscore(members[n][0], gva) for n in names}
    best_p, best_combo = -1, None
    for size in range(1, len(names) + 1):
        for combo in itertools.combinations(names, size):
            p = evaluate(uva, yva, np.mean([tva[n] for n in combo], axis=0))['primary']
            if p > best_p:
                best_p, best_combo = p, combo
    if verbose:
        print(f'  valid-selected subset: {best_combo}  valid primary={best_p:.4f}')

    key = 0 if target_split == 'valid' else 1
    _, _, u_t = enc[target_split]
    gt = _group_index(u_t)
    fused = np.mean([_zscore(members[n][key], gt) for n in best_combo], axis=0)
    return fused, best_p, best_combo


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--mode', default='single', choices=['single', 'ensemble'])
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--scores_dir', default='./scores')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print(f'loading {a.data_dir} ...')
    splits = load(a.data_dir)
    rows = splits[a.split]

    if a.mode == 'single':
        print('training validation-best single model (7 fields + BPR + watchtime aux) ...')
        m, enc, valid_best = train_best_single(splits, seed=a.seed)
        X, y, u = enc[a.split]
        scores = m.predict(X)
        print(f'  valid-best primary during training: {valid_best:.4f}')
    else:
        print('building heterogeneous ensemble from cached member scores ...')
        scores, valid_best, combo = ensemble_scores(splits, a.scores_dir, a.split)

    write_submission(a.out, rows, scores)
    print(f'wrote {a.out}: {len(rows):,d} rows (split={a.split}, mode={a.mode})')

    # 立刻自检：格式 + 对齐（跟 submit.py --check 同一套校验代码）
    back = read_submission(a.out, rows)
    print(f'✓ format/alignment check passed ({len(back):,d} rows)')
    r = evaluate([x[1] for x in rows], [x[6] for x in rows], back)
    print(f'  {a.split} GAUC {r["GAUC"]:.4f} | nDCG@5 {r["nDCG@5"]:.4f} | primary {r["primary"]:.4f}')
