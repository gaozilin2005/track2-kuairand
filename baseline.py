"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0
        # 多任务辅助头：跟主任务共享 V，但有自己的一阶项，不跟主任务的 W/b 混。
        # W_aux/b_aux：单任务时用（is_click 或 watchtime，二选一，见 pairwise_multitask /
        # pairwise_watchtime）。W_aux2/b_aux2：两个辅助任务一起训练时（pairwise_combined）
        # watchtime 专用的第二个头，跟 W_aux（此时是 click 头）分开，互不干扰。
        self.W_aux = np.zeros(dim, dtype=np.float32)
        self.b_aux = np.float32(0.0)
        self.mW_aux = np.zeros_like(self.W_aux); self.vW_aux = np.zeros_like(self.W_aux)
        self.W_aux2 = np.zeros(dim, dtype=np.float32)
        self.b_aux2 = np.float32(0.0)
        self.mW_aux2 = np.zeros_like(self.W_aux2); self.vW_aux2 = np.zeros_like(self.W_aux2)

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _adam_update(self, gV, gW, gb):
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * gb

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._adam_update(gV, gW, g.sum())
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_pairwise(self, Xpos, Xneg, weight=None):
        """BPR：loss = -log(sigmoid(s_pos - s_neg))，可选按 pair 加权（LambdaRank 用）。
        b 是全局标量，在差分中恒相消，权重不改变这一点（w 对 pos/neg 是同一个数）。"""
        B = len(Xpos)
        zpos, Epos, Spos = self.logits(Xpos)
        zneg, Eneg, Sneg = self.logits(Xneg)
        sig = sigmoid(zpos - zneg)
        w = np.ones(B, dtype=np.float32) if weight is None else weight.astype(np.float32)
        gpos = (w * (sig - 1) / B).astype(np.float32)
        gneg = -gpos
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xpos, gpos[:, None]); np.add.at(gW, Xneg, gneg[:, None])
        np.add.at(gV, Xpos, gpos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gneg[:, None, None] * (Sneg[:, None, :] - Eneg))
        self._adam_update(gV, gW, 0.0)
        return float(-np.mean(w * np.log(sig + 1e-9)))

    def step_pairwise_multitask(self, Xpos, Xneg, yclick_pos, yclick_neg, aux_weight=0.2):
        """主任务 BPR（long_view 排序，跟 step_pairwise 完全一样）+ 辅助任务
        pointwise BCE（is_click，同一批 pos/neg 行，标签更密）。两个任务共享
        V（同一个 inter），辅助任务另开 W_aux/b_aux，不跟主任务的 W/b 混。
        aux_weight 控制辅助 loss 的梯度对共享 V 的权重。
        返回 (bpr_loss, click_bce) 方便分开打印。"""
        B = len(Xpos)
        zpos, Epos, Spos = self.logits(Xpos)
        zneg, Eneg, Sneg = self.logits(Xneg)
        sig = sigmoid(zpos - zneg)
        gpos = ((sig - 1) / B).astype(np.float32)
        gneg = -gpos

        inter_pos = 0.5 * ((Spos ** 2).sum(1) - (Epos ** 2).sum((1, 2)))
        inter_neg = 0.5 * ((Sneg ** 2).sum(1) - (Eneg ** 2).sum((1, 2)))
        zclick_pos = self.b_aux + self.W_aux[Xpos].sum(1) + inter_pos
        zclick_neg = self.b_aux + self.W_aux[Xneg].sum(1) + inter_neg
        gclick_pos = (aux_weight * (sigmoid(zclick_pos) - yclick_pos) / B).astype(np.float32)
        gclick_neg = (aux_weight * (sigmoid(zclick_neg) - yclick_neg) / B).astype(np.float32)

        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xpos, gpos[:, None]); np.add.at(gW, Xneg, gneg[:, None])
        np.add.at(gV, Xpos, gpos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gneg[:, None, None] * (Sneg[:, None, :] - Eneg))
        # 辅助任务的梯度也流进同一个 V（共享 embedding）——(S-E) 结构跟主任务完全一样，
        # 只是系数换成 gclick，两路贡献直接累加进同一个 gV。
        np.add.at(gV, Xpos, gclick_pos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gclick_neg[:, None, None] * (Sneg[:, None, :] - Eneg))
        self._adam_update(gV, gW, 0.0)   # 主任务 V/W/b；b 梯度仍恒为 0（证明与 step_pairwise 相同，辅助任务不碰 b）

        # W_aux/b_aux 自己的 Adam：跟主 _adam_update 共用 self.t（已在上面那次调用里 +1），调度对齐。
        gW_aux = np.zeros_like(self.W_aux)
        np.add.at(gW_aux, Xpos, gclick_pos[:, None]); np.add.at(gW_aux, Xneg, gclick_neg[:, None])
        gW_aux += self.l2 * self.W_aux
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.mW_aux *= b1; self.mW_aux += (1 - b1) * gW_aux
        self.vW_aux *= b2; self.vW_aux += (1 - b2) * (gW_aux * gW_aux)
        self.W_aux -= self.lr * (self.mW_aux / (1 - b1 ** self.t)) / (np.sqrt(self.vW_aux / (1 - b2 ** self.t)) + eps)
        self.b_aux -= self.lr * (gclick_pos.sum() + gclick_neg.sum())

        zclick_all = np.concatenate([zclick_pos, zclick_neg])
        yclick_all = np.concatenate([yclick_pos, yclick_neg])
        click_bce = float(-np.mean(yclick_all * np.log(sigmoid(zclick_all) + 1e-9)
                                    + (1 - yclick_all) * np.log(1 - sigmoid(zclick_all) + 1e-9)))
        return float(-np.mean(np.log(sig + 1e-9))), click_bce

    def step_pairwise_watchtime(self, Xpos, Xneg, tpos, taupos, cpos, tneg, taun, cneg, aux_weight=0.2):
        """主任务 BPR + 辅助任务 CWM 风格删失回归（观看时长）。共享 V，独立
        W_aux/b_aux（跟 pairwise_multitask 用同一套头，两者不会同时训练，
        复用无冲突）。z_aux 是原始预测值，不过 sigmoid——这是回归，不是分类。

        未删失行（播到一半就走了，精确观测）：0.5*(z-t)^2，dL/dz = z-t。
        删失行（播完了，真实时长只知道 >= duration_ms 这个下界）：单侧 hinge，
        0.5*max(0, tau-z)^2，dL/dz = -max(0, tau-z)（预测已经 >= tau 时不罚，
        这正是"删失"要表达的：我们不知道真实值比 tau 高多少，不该假装知道）。
        """
        B = len(Xpos)
        zpos, Epos, Spos = self.logits(Xpos)
        zneg, Eneg, Sneg = self.logits(Xneg)
        sig = sigmoid(zpos - zneg)
        gpos = ((sig - 1) / B).astype(np.float32)
        gneg = -gpos

        inter_pos = 0.5 * ((Spos ** 2).sum(1) - (Epos ** 2).sum((1, 2)))
        inter_neg = 0.5 * ((Sneg ** 2).sum(1) - (Eneg ** 2).sum((1, 2)))
        zwt_pos = self.b_aux + self.W_aux[Xpos].sum(1) + inter_pos
        zwt_neg = self.b_aux + self.W_aux[Xneg].sum(1) + inter_neg

        gwt_pos_raw = np.where(cpos, -np.maximum(0.0, taupos - zwt_pos), zwt_pos - tpos)
        gwt_neg_raw = np.where(cneg, -np.maximum(0.0, taun - zwt_neg), zwt_neg - tneg)
        gwt_pos = (aux_weight * gwt_pos_raw / B).astype(np.float32)
        gwt_neg = (aux_weight * gwt_neg_raw / B).astype(np.float32)

        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xpos, gpos[:, None]); np.add.at(gW, Xneg, gneg[:, None])
        np.add.at(gV, Xpos, gpos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gneg[:, None, None] * (Sneg[:, None, :] - Eneg))
        np.add.at(gV, Xpos, gwt_pos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gwt_neg[:, None, None] * (Sneg[:, None, :] - Eneg))
        self._adam_update(gV, gW, 0.0)

        gW_aux = np.zeros_like(self.W_aux)
        np.add.at(gW_aux, Xpos, gwt_pos[:, None]); np.add.at(gW_aux, Xneg, gwt_neg[:, None])
        gW_aux += self.l2 * self.W_aux
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.mW_aux *= b1; self.mW_aux += (1 - b1) * gW_aux
        self.vW_aux *= b2; self.vW_aux += (1 - b2) * (gW_aux * gW_aux)
        self.W_aux -= self.lr * (self.mW_aux / (1 - b1 ** self.t)) / (np.sqrt(self.vW_aux / (1 - b2 ** self.t)) + eps)
        self.b_aux -= self.lr * (gwt_pos.sum() + gwt_neg.sum())

        loss_pos = np.where(cpos, 0.5 * np.maximum(0.0, taupos - zwt_pos) ** 2, 0.5 * (zwt_pos - tpos) ** 2)
        loss_neg = np.where(cneg, 0.5 * np.maximum(0.0, taun - zwt_neg) ** 2, 0.5 * (zwt_neg - tneg) ** 2)
        wt_loss = float(np.mean(np.concatenate([loss_pos, loss_neg])))
        return float(-np.mean(np.log(sig + 1e-9))), wt_loss

    def step_pairwise_combined(self, Xpos, Xneg, yclick_pos, yclick_neg,
                                tpos, taupos, cpos, tneg, taun, cneg,
                                click_weight=0.2, wt_weight=0.2):
        """两个辅助任务一起训练：is_click（BCE，头是 W_aux/b_aux）+ watchtime
        （删失回归，头是 W_aux2/b_aux2，独立参数，互不覆盖）。主任务 BPR 不变。
        两路辅助梯度都流进同一个共享 V，各自权重独立控制（click_weight/wt_weight）。"""
        B = len(Xpos)
        zpos, Epos, Spos = self.logits(Xpos)
        zneg, Eneg, Sneg = self.logits(Xneg)
        sig = sigmoid(zpos - zneg)
        gpos = ((sig - 1) / B).astype(np.float32)
        gneg = -gpos

        inter_pos = 0.5 * ((Spos ** 2).sum(1) - (Epos ** 2).sum((1, 2)))
        inter_neg = 0.5 * ((Sneg ** 2).sum(1) - (Eneg ** 2).sum((1, 2)))

        zclick_pos = self.b_aux + self.W_aux[Xpos].sum(1) + inter_pos
        zclick_neg = self.b_aux + self.W_aux[Xneg].sum(1) + inter_neg
        gclick_pos = (click_weight * (sigmoid(zclick_pos) - yclick_pos) / B).astype(np.float32)
        gclick_neg = (click_weight * (sigmoid(zclick_neg) - yclick_neg) / B).astype(np.float32)

        zwt_pos = self.b_aux2 + self.W_aux2[Xpos].sum(1) + inter_pos
        zwt_neg = self.b_aux2 + self.W_aux2[Xneg].sum(1) + inter_neg
        gwt_pos_raw = np.where(cpos, -np.maximum(0.0, taupos - zwt_pos), zwt_pos - tpos)
        gwt_neg_raw = np.where(cneg, -np.maximum(0.0, taun - zwt_neg), zwt_neg - tneg)
        gwt_pos = (wt_weight * gwt_pos_raw / B).astype(np.float32)
        gwt_neg = (wt_weight * gwt_neg_raw / B).astype(np.float32)

        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xpos, gpos[:, None]); np.add.at(gW, Xneg, gneg[:, None])
        np.add.at(gV, Xpos, gpos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gneg[:, None, None] * (Sneg[:, None, :] - Eneg))
        np.add.at(gV, Xpos, gclick_pos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gclick_neg[:, None, None] * (Sneg[:, None, :] - Eneg))
        np.add.at(gV, Xpos, gwt_pos[:, None, None] * (Spos[:, None, :] - Epos))
        np.add.at(gV, Xneg, gwt_neg[:, None, None] * (Sneg[:, None, :] - Eneg))
        self._adam_update(gV, gW, 0.0)

        b1, b2, eps = 0.9, 0.999, 1e-8
        gW_aux = np.zeros_like(self.W_aux)
        np.add.at(gW_aux, Xpos, gclick_pos[:, None]); np.add.at(gW_aux, Xneg, gclick_neg[:, None])
        gW_aux += self.l2 * self.W_aux
        self.mW_aux *= b1; self.mW_aux += (1 - b1) * gW_aux
        self.vW_aux *= b2; self.vW_aux += (1 - b2) * (gW_aux * gW_aux)
        self.W_aux -= self.lr * (self.mW_aux / (1 - b1 ** self.t)) / (np.sqrt(self.vW_aux / (1 - b2 ** self.t)) + eps)
        self.b_aux -= self.lr * (gclick_pos.sum() + gclick_neg.sum())

        gW_aux2 = np.zeros_like(self.W_aux2)
        np.add.at(gW_aux2, Xpos, gwt_pos[:, None]); np.add.at(gW_aux2, Xneg, gwt_neg[:, None])
        gW_aux2 += self.l2 * self.W_aux2
        self.mW_aux2 *= b1; self.mW_aux2 += (1 - b1) * gW_aux2
        self.vW_aux2 *= b2; self.vW_aux2 += (1 - b2) * (gW_aux2 * gW_aux2)
        self.W_aux2 -= self.lr * (self.mW_aux2 / (1 - b1 ** self.t)) / (np.sqrt(self.vW_aux2 / (1 - b2 ** self.t)) + eps)
        self.b_aux2 -= self.lr * (gwt_pos.sum() + gwt_neg.sum())

        zclick_all = np.concatenate([zclick_pos, zclick_neg])
        yclick_all = np.concatenate([yclick_pos, yclick_neg])
        click_bce = float(-np.mean(yclick_all * np.log(sigmoid(zclick_all) + 1e-9)
                                    + (1 - yclick_all) * np.log(1 - sigmoid(zclick_all) + 1e-9)))
        loss_pos = np.where(cpos, 0.5 * np.maximum(0.0, taupos - zwt_pos) ** 2, 0.5 * (zwt_pos - tpos) ** 2)
        loss_neg = np.where(cneg, 0.5 * np.maximum(0.0, taun - zwt_neg) ** 2, 0.5 * (zwt_neg - tneg) ** 2)
        wt_loss = float(np.mean(np.concatenate([loss_pos, loss_neg])))
        return float(-np.mean(np.log(sig + 1e-9))), click_bce, wt_loss

    def step_listwise(self, X, y, group_ids, n_groups):
        """组内 softmax 交叉熵：target 在该用户正例上均匀分布，负例 target=0。
        b 在组内 softmax 平移不变，梯度恒为 0（与 pairwise 同理）。"""
        z, E, S = self.logits(X)
        gmax = np.full(n_groups, -np.inf, dtype=np.float32)
        np.maximum.at(gmax, group_ids, z)
        ez = np.exp(z - gmax[group_ids])
        gsum = np.zeros(n_groups, dtype=np.float32)
        np.add.at(gsum, group_ids, ez)
        p = ez / gsum[group_ids]
        npos = np.zeros(n_groups, dtype=np.float32)
        np.add.at(npos, group_ids, y)
        t = y / npos[group_ids]
        g = ((p - t) / n_groups).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._adam_update(gV, gW, 0.0)
        return float(-np.sum(t * np.log(p + 1e-9)) / n_groups)

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def _listwise_epoch_batches(groups, rng, bs):
    """按整组打包成 batch，一个组永远不跨 batch（softmax 需要组内完整）。"""
    order = rng.permutation(len(groups))
    buf, buf_len = [], 0
    for oi in order:
        g = groups[oi]
        buf.append(g); buf_len += len(g)
        if buf_len >= bs:
            yield buf; buf, buf_len = [], 0
    if buf:
        yield buf

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, loss='pointwise',
           lambda_k=5, aux_weight=0.2, aux_target_train=None, wt_target_train=None,
           click_weight=None, wt_weight=None, dns_n=8, dns_warmup=3, dns_lr_decay=0.2,
           adt_beta=0.25, adt_warmup=3, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    click_weight = aux_weight if click_weight is None else click_weight
    wt_weight = aux_weight if wt_weight is None else wt_weight

    if loss in ('pairwise_multitask', 'pairwise_combined') and aux_target_train is None:
        raise ValueError(f"loss={loss!r} 需要 aux_target_train（跟 Xtr 同序的辅助标签数组，"
                          "例如 data.aux_labels(splits)['train']）")
    if loss in ('pairwise_watchtime', 'pairwise_combined') and wt_target_train is None:
        raise ValueError(f"loss={loss!r} 需要 wt_target_train=(t,tau,censored)，"
                          "例如 data.watch_time_targets(splits)['train']")

    if loss in ('pairwise', 'listwise', 'lambdarank', 'pairwise_multitask',
                'pairwise_watchtime', 'pairwise_combined', 'pairwise_dns', 'pairwise_adt'):
        # 只对有正有负的用户训练排序损失，与 GAUC 的用户口径一致（全正/全负组对组内排序无意义）。
        user_pos = collections.defaultdict(list); user_neg = collections.defaultdict(list)
        for i, (u, yv) in enumerate(zip(utr, ytr)):
            (user_pos if yv > 0 else user_neg)[u].append(i)
        mixed = [u for u in user_pos if u in user_neg]
        if loss in ('pairwise', 'lambdarank', 'pairwise_multitask', 'pairwise_watchtime',
                    'pairwise_combined', 'pairwise_dns', 'pairwise_adt'):
            # 每个正例配一个同用户负例（该用户曝光内）。
            pos_blocks = [np.array(user_pos[u]) for u in mixed]
            neg_pools = [np.array(user_neg[u]) for u in mixed]
            pos_idx_all = np.concatenate(pos_blocks)
            counts = np.array([len(b) for b in pos_blocks])
            if verbose:
                print(f"  {loss}: {len(mixed)} mixed users, {len(pos_idx_all)} pos-anchored pairs/epoch")
            if loss == 'lambdarank':
                # 排序损失只用于确定 lambda 权重，rank 每个 epoch 开始时算一次（不逐 step 重算）。
                groups = [np.concatenate([pos_blocks[i], neg_pools[i]]) for i in range(len(mixed))]
                user_of_pos = np.repeat(np.arange(len(mixed)), counts)
                idcg = np.array([sum(1.0 / np.log2(r + 1) for r in range(1, min(c, lambda_k) + 1))
                                  for c in counts])
        else:
            groups = [np.array(user_pos[u] + user_neg[u]) for u in mixed]
            if verbose:
                print(f"  listwise: {len(groups)} mixed-user groups, {sum(len(g) for g in groups)} rows/epoch")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if loss == 'pairwise':
            neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                           for pool, c in zip(neg_pools, counts)])
            perm = rng.permutation(len(pos_idx_all))
            pi, ni = pos_idx_all[perm], neg_idx_all[perm]
            losses = [m.step_pairwise(Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]]) for i in range(0, len(pi), bs)]
        elif loss == 'lambdarank':
            # rank 基于本 epoch 开始时的模型状态；epoch 内会有轻微滞后（更新一次的取舍，见 run_fm 说明）。
            score_all = m.predict(Xtr)
            row_rank = np.full(len(ytr), lambda_k + 1, dtype=np.int32)
            for grp in groups:
                order = np.argsort(-score_all[grp])
                ranks = np.empty(len(grp), dtype=np.int32)
                ranks[order] = np.arange(1, len(grp) + 1)
                row_rank[grp] = ranks
            disc_all = np.where(row_rank <= lambda_k, 1.0 / np.log2(row_rank + 1.0), 0.0)
            neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                           for pool, c in zip(neg_pools, counts)])
            perm = rng.permutation(len(pos_idx_all))
            pi, ni = pos_idx_all[perm], neg_idx_all[perm]
            lam = np.abs(disc_all[pi] - disc_all[ni]) / idcg[user_of_pos[perm]]
            losses = [m.step_pairwise(Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]], lam[i:i + bs])
                      for i in range(0, len(pi), bs)]
            if verbose:
                print(f"    lambda: mean={lam.mean():.4f} zero_frac={np.mean(lam < 1e-9):.3f}")
        elif loss == 'listwise':
            losses = []
            for batch_groups in _listwise_epoch_batches(groups, rng, bs):
                row_idx = np.concatenate(batch_groups)
                gid = np.concatenate([np.full(len(g), gi, dtype=np.int64)
                                       for gi, g in enumerate(batch_groups)])
                losses.append(m.step_listwise(Xtr[row_idx], ytr[row_idx], gid, len(batch_groups)))
        elif loss == 'pairwise_multitask':
            neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                           for pool, c in zip(neg_pools, counts)])
            perm = rng.permutation(len(pos_idx_all))
            pi, ni = pos_idx_all[perm], neg_idx_all[perm]
            step_losses = [m.step_pairwise_multitask(Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]],
                                                       aux_target_train[pi[i:i + bs]],
                                                       aux_target_train[ni[i:i + bs]],
                                                       aux_weight=aux_weight)
                           for i in range(0, len(pi), bs)]
            losses = [bpr for bpr, _ in step_losses]
            if verbose:
                print(f"    aux BCE: {np.mean([c for _, c in step_losses]):.4f} (weight={aux_weight})")
        elif loss == 'pairwise_watchtime':
            neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                           for pool, c in zip(neg_pools, counts)])
            perm = rng.permutation(len(pos_idx_all))
            pi, ni = pos_idx_all[perm], neg_idx_all[perm]
            t_arr, tau_arr, cens_arr = wt_target_train
            step_losses = [m.step_pairwise_watchtime(
                                Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]],
                                t_arr[pi[i:i + bs]], tau_arr[pi[i:i + bs]], cens_arr[pi[i:i + bs]],
                                t_arr[ni[i:i + bs]], tau_arr[ni[i:i + bs]], cens_arr[ni[i:i + bs]],
                                aux_weight=aux_weight)
                           for i in range(0, len(pi), bs)]
            losses = [bpr for bpr, _ in step_losses]
            if verbose:
                print(f"    aux watchtime loss: {np.mean([c for _, c in step_losses]):.4f} (weight={aux_weight})")
        elif loss == 'pairwise_combined':
            neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                           for pool, c in zip(neg_pools, counts)])
            perm = rng.permutation(len(pos_idx_all))
            pi, ni = pos_idx_all[perm], neg_idx_all[perm]
            t_arr, tau_arr, cens_arr = wt_target_train
            step_losses = [m.step_pairwise_combined(
                                Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]],
                                aux_target_train[pi[i:i + bs]], aux_target_train[ni[i:i + bs]],
                                t_arr[pi[i:i + bs]], tau_arr[pi[i:i + bs]], cens_arr[pi[i:i + bs]],
                                t_arr[ni[i:i + bs]], tau_arr[ni[i:i + bs]], cens_arr[ni[i:i + bs]],
                                click_weight=click_weight, wt_weight=wt_weight)
                           for i in range(0, len(pi), bs)]
            losses = [bpr for bpr, _, _ in step_losses]
            if verbose:
                print(f"    aux click BCE: {np.mean([c for _, c, _ in step_losses]):.4f} (weight={click_weight}) | "
                      f"aux watchtime loss: {np.mean([w for _, _, w in step_losses]):.4f} (weight={wt_weight})")
        elif loss == 'pairwise_adt':
            # Adaptive Denoising Training（Wang et al., WSDM 2021）的 reweighted 变体（R-CE）：
            # 噪声交互在训练早期表现为**大 loss**，所以按 loss 大小降权，而不是像 DNS 那样
            # 反过来专挑最难的样本。这正是 DNS 失败诊断的反向验证——如果"专注高 loss 样本"
            # 会让训练崩（实测如此），那"系统性地给高 loss 样本降权"就该有帮助。
            # 这个数据集的标签噪声不是假设：long_view 本质是观看时长过 18 秒阈值的离散化
            # （单这一条就能猜中 96.7%），卡在阈值附近的曝光基本等于抛硬币。
            # 权重 w = (1 - sig)^beta，sig=sigmoid(zpos-zneg) 越小（loss 越大）权重越低。
            # beta=0 退化成普通 BPR；beta 越大降权越狠。
            neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                           for pool, c in zip(neg_pools, counts)])
            perm = rng.permutation(len(pos_idx_all))
            pi, ni = pos_idx_all[perm], neg_idx_all[perm]
            losses = []
            drop_fracs = []
            for i in range(0, len(pi), bs):
                bpi, bni = pi[i:i + bs], ni[i:i + bs]
                zp = m.predict(Xtr[bpi]); zn = m.predict(Xtr[bni])
                sig = sigmoid(zp - zn)
                # 早期 epoch 不降权（模型还没学出东西，大 loss 不代表噪声，跟 DNS 的 warmup 同理）
                if ep <= adt_warmup:
                    w = None
                else:
                    w = (np.maximum(sig, 1e-6) ** adt_beta).astype(np.float32)
                    drop_fracs.append(float((w < 0.5).mean()))
                losses.append(m.step_pairwise(Xtr[bpi], Xtr[bni], weight=w))
            if verbose:
                if ep <= adt_warmup:
                    print(f"    adt: warmup epoch ({ep}/{adt_warmup}), no reweighting")
                else:
                    print(f"    adt: beta={adt_beta}, frac of pairs downweighted below 0.5 = {np.mean(drop_fracs):.3f}")
        elif loss == 'pairwise_dns':
            # Dynamic Negative Sampling：每个正例先随机抽 dns_n 个候选负例（同用户曝光内），
            # 用本 epoch 开始时的模型状态打分，挑分数最高（当前模型最容易搞错）的一个当负例。
            # rank 每个 epoch 算一次，不逐 step 重算（跟 lambdarank 的取舍一致，见其注释）。
            # 前 dns_warmup 个 epoch 先用普通随机负例——对着还没学出结构的随机 embedding
            # 挑"最难"负例等于在追噪声，实测（见 RUN_LOG.md）不warmup 会直接训练不稳定
            # （loss 不降反升）。这是 hard negative mining 文献里的标准做法，不是本项目独有的权宜之计。
            if ep <= dns_warmup:
                neg_idx_all = np.concatenate([rng.choice(pool, size=c, replace=True)
                                               for pool, c in zip(neg_pools, counts)])
                perm = rng.permutation(len(pos_idx_all))
                pi, ni = pos_idx_all[perm], neg_idx_all[perm]
                if verbose:
                    print(f"    dns: warmup epoch ({ep}/{dns_warmup}), plain random negative")
            else:
                if ep == dns_warmup + 1:
                    # 硬负例让 sigmoid(zpos-zneg) 更接近 0.5，每一步的梯度幅度都比随机负例
                    # 大得多（随机负例经常已经分对，sigmoid≈1，梯度≈0；硬负例几乎每步都有实质
                    # 梯度）——相当于变相调高了有效学习率，Adam 的一阶/二阶矩估计跟不上就会
                    # 震荡（实测：不降 lr 直接 loss 不降反升，见 RUN_LOG.md）。切换时降一次 lr，
                    # 是 hard negative mining 文献里的标准做法。
                    m.lr *= dns_lr_decay
                    if verbose:
                        print(f"    dns: switching to hard negatives, lr *= {dns_lr_decay} -> {m.lr:.6f}")
                cand_idx = np.concatenate([rng.choice(pool, size=(c, dns_n), replace=True)
                                            for pool, c in zip(neg_pools, counts)])   # (n_pos, dns_n)
                cand_scores = m.predict(Xtr[cand_idx.reshape(-1)]).reshape(cand_idx.shape)
                hardest = cand_idx[np.arange(len(cand_idx)), cand_scores.argmax(axis=1)]
                perm = rng.permutation(len(pos_idx_all))
                pi, ni = pos_idx_all[perm], hardest[perm]
                if verbose:
                    print(f"    dns: mean hardest-candidate score={cand_scores.max(axis=1).mean():.4f} "
                          f"vs mean candidate score={cand_scores.mean():.4f} (n={dns_n})")
            losses = [m.step_pairwise(Xtr[pi[i:i + bs]], Xtr[ni[i:i + bs]]) for i in range(0, len(pi), bs)]
        else:
            idx = rng.permutation(len(ytr))
            losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loss', default='pairwise_watchtime',
                    choices=['pointwise', 'pairwise', 'listwise', 'lambdarank',
                             'pairwise_multitask', 'pairwise_watchtime', 'pairwise_combined', 'pairwise_dns',
                             'pairwise_adt'],
                    help='仅对 --model fm 生效：pairwise_watchtime（BPR + CWM 风格观看时长删失回归辅助任务，'
                         '默认，5-seed test primary 0.6017 > 纯 BPR（7 域）的 0.6008，见 RUN_LOG.md）/ '
                         'pairwise（BPR，无辅助任务）/ pointwise（logloss，原官方 baseline）/ '
                         'listwise（组内 softmax，已实测更差）/ lambdarank（BPR × |ΔnDCG@lambda_k|，已实测更差）/ '
                         'pairwise_multitask（BPR + is_click 辅助 BCE，已实测无收益）/ '
                         'pairwise_combined（BPR + is_click + watchtime 两个辅助任务一起训练，各自独立头）/ '
                         'pairwise_dns（Dynamic Negative Sampling：每个正例从 dns_n 个候选负例里挑当前模型打分最高的一个）')
    ap.add_argument('--lambda_k', type=int, default=5, help='lambdarank 用的截断位置，默认对齐 nDCG@5')
    ap.add_argument('--aux_weight', type=float, default=0.2,
                    help='pairwise_multitask / pairwise_watchtime 用：辅助任务 loss 对共享 V 的梯度权重')
    ap.add_argument('--click_weight', type=float, default=None,
                    help='pairwise_combined 用：is_click 头的权重，不传则用 --aux_weight')
    ap.add_argument('--wt_weight', type=float, default=None,
                    help='pairwise_combined 用：watchtime 头的权重，不传则用 --aux_weight')
    ap.add_argument('--dns_n', type=int, default=8,
                    help='pairwise_dns 用：每个正例采样的候选负例个数，从中选分数最高的一个')
    ap.add_argument('--dns_warmup', type=int, default=3,
                    help='pairwise_dns 用：前几个 epoch 用普通随机负例，之后再切到 hard negative')
    ap.add_argument('--dns_lr_decay', type=float, default=0.2,
                    help='pairwise_dns 用：切到 hard negative 时 lr 乘的系数，避免梯度幅度突增导致震荡')
    ap.add_argument('--adt_beta', type=float, default=0.25,
                    help='pairwise_adt 用：降权强度，w=(sigmoid(zpos-zneg))^beta，0 等于不降权')
    ap.add_argument('--adt_warmup', type=int, default=3,
                    help='pairwise_adt 用：前几个 epoch 不降权（早期大 loss 不代表噪声）')
    ap.add_argument('--wt_target', default='log', choices=['log', 'quantile'],
                    help='pairwise_watchtime / pairwise_combined 用：辅助任务的目标形式。'
                         'log=截断+log1p 回归（默认，当前最优 0.6017 用的就是这个）；'
                         'quantile=RAD（AAAI 2025）风格，回归观看时长在同时长组经验分布里的分位数')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    aux_train, wt_train = None, None
    if a.loss in ('pairwise_multitask', 'pairwise_combined'):
        from data import aux_labels
        aux_train = aux_labels(splits)['train']
    if a.loss in ('pairwise_watchtime', 'pairwise_combined'):
        if a.wt_target == 'quantile':
            from data import watch_time_quantile_targets
            wt_train = watch_time_quantile_targets(splits)['train']
        else:
            from data import watch_time_targets
            wt_train = watch_time_targets(splits)['train']
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                                   loss=a.loss, lambda_k=a.lambda_k, aux_weight=a.aux_weight,
                                   aux_target_train=aux_train, wt_target_train=wt_train,
                                   click_weight=a.click_weight, wt_weight=a.wt_weight,
                                   dns_n=a.dns_n, dns_warmup=a.dns_warmup,
                                   dns_lr_decay=a.dns_lr_decay,
                                   adt_beta=a.adt_beta, adt_warmup=a.adt_warmup)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}, loss={a.loss}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
