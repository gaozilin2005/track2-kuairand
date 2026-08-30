"""学习式集成（stacking）：让一个小模型学怎么组合各成员的分数，而不是手工挑一个
固定的融合方式。

跟 `ablation_hetero_ensemble.py` 的关系：那边试过的 raw / z-score / rank / RRF /
贪心加权，全部是**全局固定权重**——不管是哪个用户、哪一行，同一个成员永远用同一个
权重组合。文献里 stacked generalization（Wolpert 1992）的核心论点是：一个学出来的
元模型可以把"哪个成员更可信"这件事**跟输入本身挂钩**——比如误差不再是全局同分布，
而是"BST 在历史丰富的用户上更准，FM 在冷用户上更准"这种依情况而定的模式，固定权重
天生学不到这个，学出来的组合器可以。

做法：把 6 个成员（bst / deepfm / finalmlp / fm_quantile / fm_watchtime / lightgcn）
在某一行的分数当作 6 维输入特征，喂给一个小模型，用跟本项目其它地方完全一致的
BPR pairwise 目标训练（不是普通的 pointwise 分类——这是排序任务，训练目标要跟
评测指标对齐）。

**防止过拟合 valid 的关键设计**：跟 `ablation_hetero_ensemble.py` 一样"选择只在
valid 上做、test 只报一次"这个规矩不够了——这次不是从几个离散方案里挑一个，而是
真的在拟合参数，valid 用完就有信息泄漏的风险。所以把 valid 按用户切成
meta_train / meta_dev 两半：在 meta_train 上训参数，在 meta_dev 上早停/选模型，
最后只在 test 上报一次——test 全程不参与任何决策。

两个模型都试：
  linear  等价于给 6 个成员学一个全局最优的线性组合（比固定的 z-score 等权已经
          多一层自由度——例如可以学出"BST 权重该比看起来更大"）
  mlp     6→16→1 的小 MLP，能表达"跟成员之间关系相关"的组合方式，是否比线性更好
          直接看 meta_dev 结果，不是假设它一定更强。
"""
import argparse, collections
import numpy as np
import torch
import torch.nn as nn

from data import load, encode
from evaluate import evaluate

MEMBERS = ['bst', 'deepfm', 'finalmlp', 'fm_quantile', 'fm_watchtime', 'lightgcn']


def load_members(scores_dir='./scores'):
    out = {}
    for name in MEMBERS:
        d = np.load(f'{scores_dir}/{name}.npz')
        out[name] = (d['valid'].astype(np.float32), d['test'].astype(np.float32))
    return out


def zscore_by_group(scores, groups):
    out = np.empty(len(scores), dtype=np.float32)
    for idxs in groups:
        s = scores[idxs]
        sd = s.std()
        out[idxs] = (s - s.mean()) / (sd if sd > 1e-9 else 1.0)
    return out


def group_index(users):
    by = collections.defaultdict(list)
    for i, u in enumerate(users):
        by[u].append(i)
    return [np.array(v) for v in by.values()]


class LinearStacker(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_in) / n_in)   # 从等权开始，不从随机点开始
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x @ self.w + self.b


class MLPStacker(nn.Module):
    def __init__(self, n_in, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.PReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_pairs(users, y, rng):
    """跟 baseline.py 完全一样的构造：同用户内 pos-anchored 配对。"""
    user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
    for i, (u, yv) in enumerate(zip(users, y)):
        (user_pos if yv > 0 else user_neg)[u].append(i)
    mixed = [u for u in user_pos if u in user_neg]
    pos_blocks = [np.array(user_pos[u]) for u in mixed]
    neg_pools = [np.array(user_neg[u]) for u in mixed]
    pos_idx_all = np.concatenate(pos_blocks)
    counts = np.array([len(b) for b in pos_blocks])
    neg_idx_all = np.concatenate([rng.choice(p, size=c, replace=True)
                                   for p, c in zip(neg_pools, counts)])
    return pos_idx_all, neg_idx_all


def train_stacker(model, X_tr, u_tr, y_tr, X_dev, u_dev, y_dev, epochs=200, lr=0.01, patience=15, l1=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(X_tr)
    rng = np.random.default_rng(0)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        pi, ni = make_pairs(u_tr, y_tr, rng)
        zp = model(Xtr_t[pi]); zn = model(Xtr_t[ni])
        loss = -torch.log(torch.sigmoid(zp - zn) + 1e-9).mean()
        if l1 > 0 and hasattr(model, 'w'):
            loss = loss + l1 * model.w.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            dev_scores = model(torch.from_numpy(X_dev)).numpy()
        r = evaluate(u_dev, y_dev, dev_scores)
        if r['primary'] > best + 1e-5:
            best, bad = r['primary'], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--scores_dir', default='./scores')
    ap.add_argument('--meta_train_frac', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    splits = load(a.data_dir)
    enc, _ = encode(splits)
    _, yva, uva = enc['valid']; _, yte, ute = enc['test']
    members = load_members(a.scores_dir)
    print(f'members: {list(members)}')

    gva = group_index(uva)
    Xva = np.stack([zscore_by_group(members[n][0], gva) for n in MEMBERS], axis=1)
    gte = group_index(ute)
    Xte = np.stack([zscore_by_group(members[n][1], gte) for n in MEMBERS], axis=1)

    # 按用户切 meta_train / meta_dev，不按行切——同一用户的行不能一半在 train 一半在 dev
    # （不然 dev 上"同用户内排序"这个任务本身就不完整）。
    rng = np.random.default_rng(a.seed)
    uniq_users = np.array(sorted(set(uva)))
    rng.shuffle(uniq_users)
    n_tr = int(len(uniq_users) * a.meta_train_frac)
    tr_users, dev_users = set(uniq_users[:n_tr]), set(uniq_users[n_tr:])
    tr_mask = np.array([u in tr_users for u in uva])
    dev_mask = ~tr_mask
    print(f'meta_train: {tr_mask.sum()} rows / {len(tr_users)} users | '
          f'meta_dev: {dev_mask.sum()} rows / {len(dev_users)} users')

    X_tr, u_tr, y_tr = Xva[tr_mask], [u for u, m in zip(uva, tr_mask) if m], yva[tr_mask]
    X_dev, u_dev, y_dev = Xva[dev_mask], [u for u, m in zip(uva, dev_mask) if m], yva[dev_mask]

    print('\n=== reference: equal-weight z-score average (no learning) ===')
    r = evaluate(uva, yva, Xva.mean(1))
    print(f'  valid  primary {r["primary"]:.4f}')
    r = evaluate(ute, yte, Xte.mean(1))
    print(f'  test   primary {r["primary"]:.4f}')

    torch.manual_seed(a.seed)
    configs = [('linear', LinearStacker, 0.0), ('mlp', MLPStacker, 0.0),
               ('linear_l1_0.02', LinearStacker, 0.02), ('linear_l1_0.05', LinearStacker, 0.05),
               ('linear_l1_0.1', LinearStacker, 0.1)]
    for name, Model, l1 in configs:
        model = Model(len(MEMBERS))
        model, dev_best = train_stacker(model, X_tr, u_tr, y_tr, X_dev, u_dev, y_dev, l1=l1)
        with torch.no_grad():
            va_full = evaluate(uva, yva, model(torch.from_numpy(Xva)).numpy())
            te_full = evaluate(ute, yte, model(torch.from_numpy(Xte)).numpy())
        print(f'\n=== {name} stacker ===')
        if 'linear' in name:
            w = model.w.detach().numpy()
            print('  learned weights: ' + ', '.join(f'{n}={v:+.3f}' for n, v in zip(MEMBERS, w))
                  + f', bias={model.b.item():+.3f}')
        print(f'  meta_dev primary during selection: {dev_best:.4f}')
        print(f'  valid (full, incl. meta_train)  GAUC {va_full["GAUC"]:.4f} | '
              f'nDCG@5 {va_full["nDCG@5"]:.4f} | primary {va_full["primary"]:.4f}')
        print(f'  test                             GAUC {te_full["GAUC"]:.4f} | '
              f'nDCG@5 {te_full["nDCG@5"]:.4f} | primary {te_full["primary"]:.4f}')

    print('\n  vs static-weight ensemble (RUN_LOG.md): valid 0.6091 / test 0.6034')
