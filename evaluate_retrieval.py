"""全库检索的评测指标：NDCG@10 / Recall@50，正例 = is_click。

跟 evaluate.py 的关系：**这是另一个任务，不是它的改版。** evaluate.py 实现的是
"用户内排序"（只对该用户在评测集里的曝光排序，不做全库检索，指标 GAUC/nDCG@5，
正例 long_view）。这里实现的是**全库检索**：对每个用户给整个视频库打分排序，看
用户真正点击的视频有多少落进 top-10 / top-50。

为什么必须是全库检索（不是对曝光重排）：test 里每个用户中位数只有 5 条曝光，
只有 0.3% 的用户有 ≥50 条。如果在曝光内部算 Recall@50，99.7% 的用户恒等于 1.0，
指标完全退化；NDCG@10 同理（76% 的用户曝光数不足 10）。这两个指标只有在
"从 7,583 个视频里检索"的语境下才有意义。

口径约定（**这些是实现选择，不是数据给定的，换口径分数会变，对比前先确认一致**）：
1. 候选集 = 视频库全部 7,583 个视频（video_features_basic_pure.csv 里的全集）。
2. 正例 = 该用户在评测集里 is_click=1 的视频。
3. **训练期交互过的视频从候选集里剔除**（标准检索评测做法：否则模型只要把用户
   已经看过的推回去就能拿高分，衡量不到泛化）。剔除的是 train 里出现过的
   (user, video) 对，不管当时点没点。
4. 零正例用户（评测集里一次都没点）不计入平均——跟 evaluate.py 对 GAUC 的处理
   一致，这类用户的检索指标恒为 0，留着只会稀释所有模型的分数、不改变排名。
5. NDCG 的 IDCG 用 min(正例数, k) 个理想位置算，正例数超过 k 时不惩罚。

⚠️ 约定 3 的一个副作用：评测集里有 1.2% 的正例，用户在 train 期间也交互过同一个视频，
于是被剔除出候选集 → **永远检索不到，但仍计入分母**。所以 Recall@50 的理论上限不是
1.0 而是约 0.988。这个偏差对所有模型一视同仁，不影响模型之间的排名；但如果换成
"不剔除训练期交互"的口径，绝对分数会明显不同，跨口径不可比。
"""
import numpy as np


def _dcg_weights(k):
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate_retrieval(score_matrix, user_pos, exclude=None, ks=(10, 50), batch=512):
    """
    score_matrix : (n_users, n_items) 打分，或一个 callable(user_slice)->(b, n_items)
    user_pos     : list[set[int]]，每个用户的正例 item 索引集合
    exclude      : list[set[int]] 或 None，每个用户要从候选集剔除的 item（训练期交互过的）
    返回 {'NDCG@10':…, 'Recall@50':…, 'users':…}
    """
    n_users = len(user_pos)
    kmax = max(ks)
    w = {k: _dcg_weights(k) for k in ks}
    sums = {f'NDCG@{k}': 0.0 for k in ks}
    sums.update({f'Recall@{k}': 0.0 for k in ks})
    n_scored = 0

    for start in range(0, n_users, batch):
        stop = min(start + batch, n_users)
        S = score_matrix(slice(start, stop)) if callable(score_matrix) \
            else np.asarray(score_matrix[start:stop], dtype=np.float32).copy()

        for r in range(stop - start):
            u = start + r
            pos = user_pos[u]
            if not pos:
                continue                      # 零正例用户不计入（见文件头约定 4）
            if exclude is not None and exclude[u]:
                ex = np.fromiter(exclude[u], dtype=np.int64, count=len(exclude[u]))
                S[r, ex] = -np.inf            # 训练期交互过的不参与排序
            top = np.argpartition(-S[r], kmax - 1)[:kmax]
            top = top[np.argsort(-S[r][top])]
            hit = np.fromiter((int(i in pos) for i in top), dtype=np.float64, count=kmax)
            npos = len(pos)
            for k in ks:
                h = hit[:k]
                dcg = float((h * w[k]).sum())
                idcg = float(w[k][:min(npos, k)].sum())
                sums[f'NDCG@{k}'] += dcg / idcg if idcg > 0 else 0.0
                sums[f'Recall@{k}'] += h.sum() / npos
            n_scored += 1

    out = {m: (v / n_scored if n_scored else 0.0) for m, v in sums.items()}
    out['users'] = n_scored
    return out


def build_eval_sets(splits, eval_split, item_index, label_idx=8):
    """从 data.load() 的 splits 构造检索评测需要的结构。
    label_idx=8 是 is_click（本任务的正例定义）；6 是 long_view。
    返回 (user_ids, user_pos, exclude)，三者按同一个用户顺序对齐。"""
    users = sorted({x[1] for x in splits[eval_split]})
    uidx = {u: i for i, u in enumerate(users)}
    user_pos = [set() for _ in users]
    exclude = [set() for _ in users]

    for x in splits[eval_split]:
        if x[label_idx]:
            v = item_index.get(x[2])
            if v is not None:
                user_pos[uidx[x[1]]].add(v)
    for x in splits['train']:                 # 训练期交互过的一律剔除（约定 3）
        i = uidx.get(x[1])
        if i is not None:
            v = item_index.get(x[2])
            if v is not None:
                exclude[i].add(v)
    return users, user_pos, exclude
