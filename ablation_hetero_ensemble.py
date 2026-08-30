"""异构集成：把**不同架构/不同损失**的模型的预测合起来，而不是只合并同一模型的不同 seed。

动机：ablation_ensemble.py 的同构集成只有 +0.0007（约 1.8σ）且 3 个模型就饱和——这正是
集成理论预测的结果：同构成员的误差高度相关，能削减的方差有限。文献（heterogeneous
ensemble）一致指出收益来自**成员误差模式互补**，而本项目恰好训了好几个归纳偏置完全不同、
单独分数却都在 0.600~0.602 的模型：

  fm_watchtime  0.6017   FM + 观看时长辅助任务（当前默认）
  fm_quantile   0.6016   同上，但辅助目标换成 RAD 分位数
  bst           0.6014   自注意力 + 位置编码（唯一真正用到顺序的）
  deepfm        0.6007   FM + DNN 分支
  finalmlp      0.6002   完全没有显式交互项，两路 MLP + 门控

这些模型此前只被当作"各自失败的方向"逐一否决，从没被合起来用过。BST 尤其值得进集成：
它是唯一从顺序维度提取信号的，误差模式最可能跟 FM 系互补。

聚合方式测四种（文献里的标准选项）：
  raw   —— 直接平均原始分（尺度敏感）
  zscore—— 每个模型先在组内标准化再平均（消除尺度差异）
  rank  —— 每个模型先转组内 rank 再平均（只保留序，最鲁棒）
  rrf   —— Reciprocal Rank Fusion，1/(k+rank)，IR 领域的标准做法，压低长尾名次的影响

**权重/方案的选择只用 valid，最终只在 test 上报一次**——避免在 test 上调参。
"""
import argparse, collections, itertools, statistics
import numpy as np
from data import load, encode, watch_time_targets, watch_time_quantile_targets, FIELDS
from evaluate import evaluate
import baseline as B


def train_fm(splits, enc, dim, wt, seed, epochs=40, bs=8192, patience=4,
             k=16, lr=0.001, aux_weight=0.2):
    """numpy FM + BPR + watchtime 辅助任务，返回 (valid_scores, test_scores)。"""
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
                t_arr[bni], tau_arr[bni], cens_arr[bni], aux_weight=aux_weight)
        va = evaluate(uva, yva, m.predict(Xva))['primary']
        if va > best + 1e-5:
            best, bad = va, 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return m.predict(Xva), m.predict(Xte)


# ---------- 聚合方式 ----------
def _group_index(users):
    by = collections.defaultdict(list)
    for i, u in enumerate(users):
        by[u].append(i)
    return [np.array(v) for v in by.values()]


def to_rank(scores, groups):
    """组内 rank，归一化到 [0,1]，越大越好。"""
    out = np.empty(len(scores))
    for idxs in groups:
        s = scores[idxs]
        order = np.argsort(s)
        r = np.empty(len(s)); r[order] = np.arange(len(s))
        out[idxs] = r / max(len(s) - 1, 1)
    return out


def to_zscore(scores, groups):
    out = np.empty(len(scores))
    for idxs in groups:
        s = scores[idxs]
        sd = s.std()
        out[idxs] = (s - s.mean()) / (sd if sd > 1e-9 else 1.0)
    return out


def to_rrf(scores, groups, k_rrf=60.0):
    """Reciprocal Rank Fusion：1/(k+rank)，rank 从 1 开始（分数最高的是 rank 1）。"""
    out = np.empty(len(scores))
    for idxs in groups:
        s = scores[idxs]
        order = np.argsort(-s)
        r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
        out[idxs] = 1.0 / (k_rrf + r)
    return out


TRANSFORMS = {'raw': lambda s, g: s, 'zscore': to_zscore, 'rank': to_rank, 'rrf': to_rrf}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--member', default=None,
                    help='只训练这一个成员并把分数存到 scores/<name>.npz（分开跑，避免同时驻留爆内存）')
    ap.add_argument('--seed', type=int, default=None,
                    help='覆盖该成员默认的训练种子（fm_watchtime=0/fm_quantile=1/其它=0）。'
                         '只用于多 seed 稳定性检查——不传就是原来的固定值，不影响任何已发表的数字。')
    ap.add_argument('--combine', action='store_true', help='读取 scores/ 下所有成员，做融合与评估')
    ap.add_argument('--scores_dir', default='./scores')
    ap.add_argument('--n_greedy', type=int, default=15,
                    help='贪心加权集成的迭代轮数（每轮选一个成员，可重复选）')
    ap.add_argument('--bst_L', type=int, default=64,
                    help='BST 成员的历史窗口长度。默认 160 会把内存打爆（本机 OOM，exit 137）；'
                         '64 足够——RUN_LOG 记录 L=100 也只影响 0.18% 的行')
    a = ap.parse_args()

    import os
    os.makedirs(a.scores_dir, exist_ok=True)
    splits = load(a.data_dir)
    enc, dim = encode(splits)
    Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    if a.member:
        name = a.member
        if name == 'fm_watchtime':
            sva, ste = train_fm(splits, enc, dim, watch_time_targets(splits), seed=a.seed if a.seed is not None else 0)
        elif name == 'fm_quantile':
            sva, ste = train_fm(splits, enc, dim, watch_time_quantile_targets(splits), seed=a.seed if a.seed is not None else 1)
        elif name == 'deepfm':
            from deepfm_model import run_deepfm
            r = run_deepfm(splits, seed=a.seed if a.seed is not None else 0, device=a.device, verbose=False, return_scores=True)
            sva, ste = r['valid_scores'], r['test_scores']
        elif name == 'finalmlp':
            from finalmlp_model import run_finalmlp
            r = run_finalmlp(splits, seed=a.seed if a.seed is not None else 0, device=a.device, verbose=False, return_scores=True)
            sva, ste = r['valid_scores'], r['test_scores']
        elif name == 'lightgcn':
            from lightgcn_model import run_lightgcn
            r = run_lightgcn(splits, seed=a.seed if a.seed is not None else 0, device=a.device, verbose=False, return_scores=True)
            sva, ste = r['valid_scores'], r['test_scores']
        elif name == 'bst':
            from sequence_model import run_seq
            r = run_seq(splits, arch='bst', seed=a.seed if a.seed is not None else 0, device=a.device, L=a.bst_L,
                        verbose=False, return_scores=True)
            sva, ste = r['valid_scores'], r['test_scores']
        else:
            raise ValueError(f'unknown member {name!r}')
        np.savez_compressed(os.path.join(a.scores_dir, f'{name}.npz'), valid=sva, test=ste)
        print(f'{name}: test primary {evaluate(ute, yte, ste)["primary"]:.4f}  -> saved')
        raise SystemExit

    if not a.combine:
        raise SystemExit('pass --member <name> to train one, or --combine to fuse saved scores')

    members = {}
    for fn in sorted(os.listdir(a.scores_dir)):
        if fn.endswith('.npz'):
            d = np.load(os.path.join(a.scores_dir, fn))
            members[fn[:-4]] = (d['valid'], d['test'])
    names = list(members)
    n_greedy = a.n_greedy
    print(f'loaded members: {names}')
    gva, gte = _group_index(uva), _group_index(ute)

    print('\n  individual test primary:')
    for n in names:
        print(f'    {n:14s} {evaluate(ute, yte, members[n][1])["primary"]:.4f}')

    print('\n  member correlation on test (groupwise-rank space; lower = more complementary):')
    ranks_te = {n: to_rank(members[n][1], gte) for n in names}
    print('              ' + '  '.join(f'{n[:10]:>10s}' for n in names))
    for n1 in names:
        row = '  '.join(f'{np.corrcoef(ranks_te[n1], ranks_te[n2])[0,1]:10.3f}' for n2 in names)
        print(f'  {n1[:10]:>10s} {row}')

    best_cfg, best_va = None, -1
    for tname, tf in TRANSFORMS.items():
        tva = {n: tf(members[n][0], gva) for n in names}
        for size in range(2, len(names) + 1):
            for combo in itertools.combinations(names, size):
                p = evaluate(uva, yva, np.mean([tva[n] for n in combo], axis=0))['primary']
                if p > best_va:
                    best_va, best_cfg = p, (tname, combo)
    tname, combo = best_cfg
    print(f'\n  best on VALID: transform={tname}, members={combo}, valid primary={best_va:.4f}')

    print('\n  === TEST ===')
    if 'fm_watchtime' in members:
        r = evaluate(ute, yte, members['fm_watchtime'][1])
        print(f'  single best member (fm_watchtime): GAUC {r["GAUC"]:.4f} | nDCG@5 {r["nDCG@5"]:.4f} | primary {r["primary"]:.4f}')
    tf = TRANSFORMS[tname]
    sel = evaluate(ute, yte, np.mean([tf(members[n][1], gte) for n in combo], axis=0))
    print(f'  valid-selected ensemble          : GAUC {sel["GAUC"]:.4f} | nDCG@5 {sel["nDCG@5"]:.4f} | primary {sel["primary"]:.4f}')

    print('\n  (reference) all-members ensemble under each transform:')
    for tn, tfun in TRANSFORMS.items():
        r = evaluate(ute, yte, np.mean([tfun(members[n][1], gte) for n in names], axis=0))
        print(f'    {tn:7s}: primary {r["primary"]:.4f}')

    # ---- 贪心加权集成（Caruana et al. 2004, ensemble selection with replacement）----
    # 等权平均对弱成员太不公平：LightGCN 跟其它成员的相关只有 0.57（比 BST 的 0.89 更
    # 互补），但它单独只有 0.5576，等权投票会把整体拖下去。带放回的贪心选择天然给出
    # 整数权重——一个成员被选中几次就是几份权重，弱而互补的成员可以只拿一小份。
    # 仍然只在 valid 上选，test 只在最后报一次。
    print('\n  ---- greedy weighted ensemble (selection with replacement, on VALID) ----')
    for tname, tf in TRANSFORMS.items():
        tva = {n: tf(members[n][0], gva) for n in names}
        tte = {n: tf(members[n][1], gte) for n in names}
        counts = {n: 0 for n in names}
        cur_va = None
        best_hist = []
        for step in range(n_greedy):
            best_p, best_n = -1, None
            for n in names:
                cand = tva[n] if cur_va is None else (cur_va * step + tva[n]) / (step + 1)
                p_ = evaluate(uva, yva, cand)['primary']
                if p_ > best_p:
                    best_p, best_n = p_, n
            counts[best_n] += 1
            cur_va = tva[best_n] if cur_va is None else (cur_va * step + tva[best_n]) / (step + 1)
            best_hist.append(best_p)
        tot = sum(counts.values())
        wstr = ', '.join(f'{n}:{c}/{tot}' for n, c in counts.items() if c)
        ens_te = sum(counts[n] * tte[n] for n in names) / tot
        r = evaluate(ute, yte, ens_te)
        print(f'    {tname:7s}: valid {best_hist[-1]:.4f} -> TEST primary {r["primary"]:.4f}  '
              f'(GAUC {r["GAUC"]:.4f}, nDCG@5 {r["nDCG@5"]:.4f})  weights: {wstr}')

    print('\n  vs homogeneous seed ensemble (RUN_LOG.md): 0.6025 | single-model best 0.6017')
