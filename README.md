# KuaiRand-Pure Starter Kit

## 依赖

Python 3.9+ 和 numpy。**核心 baseline 没有别的。** 不需要 torch、pandas、sklearn。

`sequence_model.py`（DIN 风格用户历史 attention，见下方「从哪里开始改」）额外依赖
PyTorch，装在本地 `.venv/`（`python3 -m venv .venv && .venv/bin/pip install torch`），
不影响 `baseline.py` 本身——那条路径依然只用 numpy，未改动。

## 数据

从 https://kuairand.com 下载（Zenodo 直链，无需注册）：

```bash
# 在 Starter Kit 目录下执行，解压后得到 ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## 运行

```bash
python3 baseline.py --model fm
```

`--data_dir` 默认 `./KuaiRand-Pure/data`；数据放在别处时显式指定。

`--model` 可选 `fm`（起步模型）/ `pop`（trivial baseline）/ `random`（下界，用于自检评测代码）。
FM 全程约 40 秒（CPU，单核）。

`--loss` 默认 `pairwise_watchtime`（BPR 主任务 + CWM 风格观看时长删失回归辅助任务）——
实测比纯 BPR 好（+0.0009 primary，5 seed 均稳定但幅度比其它几步小，见 `RUN_LOG.md`
的显著性讨论）。传 `--loss pairwise` 关掉辅助任务，回到纯 BPR；传 `--loss pointwise`
用回原官方 baseline 的 loss（但不会精确复现 `baseline_scores.json` 的历史数字，见下）。

`data.py` 的 `FIELDS` 除了原本 5 个静态域，还默认带 `prior_exposure`（是否精确复看过
这个视频）和 `author_recency`（离上次看这个作者的作品过了多久，分桶）两个跨行时序特征
（2026-08-28 收编，见 `RUN_LOG.md`），同样有稳定收益（+0.0037）。这两个特征目前没有开关，
无法单独关掉——`--loss pointwise` 不再能精确复现 `baseline_scores.json` 里记录的原始数字
（那是 5 域 + pointwise 的历史快照，`baseline_scores.json` 本身保持不动）；当前默认配置的
完整数字见下面的 Baseline 阶梯，逐条演变过程见 `RUN_LOG.md`。

## 任务定义（口径已写死，不要改）

| | |
|---|---|
| 任务 | **用户内排序** —— 每个用户只对其在评测集中的曝光排序，不做全库检索 |
| 相关性标签 | `long_view`（原生列，0/1） |
| 指标 | `GAUC`、`nDCG@5`；**主分 = 两者平均** |
| 数据划分 | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| 零正例用户 | nDCG 记 0.0 并计入平均；GAUC 只统计 `0 < 正例数 < 曝光数` 的用户，按正例数加权 |
| nDCG gain | `2^rel − 1`（二元标签下等价于 identity） |

实现见 `evaluate.py`，全部约定写在文件头注释里。

## Baseline 阶梯

test 集上的分数。**要打败的是最后一行（当前默认配置）。**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random（下界，自检用） | 0.4996 | 0.4511 | 0.4753 |
| item popularity（trivial） | 0.6308 | 0.5121 | 0.5715 |
| FM + pointwise，5 域（原官方 baseline，`baseline_scores.json` 里的历史快照） | 0.6610 | 0.5282 | 0.5946 |
| FM + BPR，5 域 | 0.6638 | 0.5304 | 0.5971 |
| FM + BPR，7 域（+ `prior_exposure` + `author_recency`） | 0.6689 | 0.5326 | 0.6008 |
| **FM + BPR + CWM 风格观看时长辅助任务，7 域（当前默认）** | **0.6702** | **0.5333** | **0.6017** |

前三步提升都超过收敛阈值 ε=0.002，5 个 seed 上稳定复现：损失函数从 pointwise 换成
BPR（+0.0025，另外试过 listwise/LambdaRank@5，均更差）、再加两个时序特征（+0.0037，
组合收益略高于两者单独收益之和）。最后一步（+0.0009）幅度更小，但 5 个 seed 稳定，
用标准误差算显著性约 3.6σ（`RUN_LOG.md` 有完整讨论）；同样试过 `is_click` 辅助任务，
无收益。完整演变过程和每一步的具体实验设计见 `RUN_LOG.md`。

### ⚠️ 指标的真实区间：nDCG@5 的天花板是 0.729，不是 1.0

test 集 23,875 个用户里：

| | 占比 | 对指标的影响 |
|---|---|---|
| 全负用户（该用户所有曝光都不是 long_view） | **27.1%** | nDCG 恒为 **0**，任何模型都救不了；不计入 GAUC |
| 全正用户 | **9.2%** | nDCG 恒为 **1**；不计入 GAUC |
| 有区分度的用户 | **63.7%** | GAUC 的实际样本 |

所以用真实标签当预测分（oracle，完美排序）也只能拿到：

| | random | FM baseline | **oracle 上限** | FM 已吃掉的区间 |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**评估进展请以 oracle 为分母。** 看到 0.5946 就以为「离满分 1.0 还很远」是误判——
baseline 已经吃掉可用区间的三成，剩余 headroom 是 0.27 而不是 0.41。

（这张表用的是 `baseline_scores.json` 里的历史 5 域 pointwise 数字，方便对照官方记录；
按当前默认配置的 0.6017 算，剩余 headroom 是 0.8645 − 0.6017 = **0.26**。）

FM 在 5 个随机种子上的 std 均为 **0.0008**。据此收敛判据取 **ε = 0.002（≈2.5σ）, N = 3**：
连续 3 轮迭代 validation 主分提升不超过 0.002 即判定收敛。

> 自检：如果你的评测代码跑 `--model random` 得不到 primary ≈ 0.475（±0.001），说明 harness 有问题，先修它。

## 提交格式

CSV，含表头，一行对应评测集的一行：

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| 字段 | 说明 |
|---|---|
| `row_id` | 0 起连续递增，对应 `data.load()[split]` 的行序（确定性：先读 `log_standard_4_08_to_4_21_pure.csv` 再读 `log_standard_4_22_to_5_08_pure.csv`，按 date 过滤后保持原文件顺序） |
| `user_id` / `video_id` | 冗余字段，仅用于校验对齐 |
| `score` | 你的模型给该行打的分，任意实数，只用相对大小；不允许 NaN / Inf |

> **为什么必须带 `row_id`：** `(user_id, video_id)` 在评测集里**不唯一** ——
> test 集有 3.06% 的重复对，最多重复 12 次。所以它不能作为主键。

生成与校验：

```bash
python3 submit.py --make  --split test  submission.csv    # 用官方 FM baseline 生成一份示例提交
python3 submit.py --check --split test  submission.csv    # 校验格式与对齐
python3 submit.py --score --split valid submission.csv    # 校验并打分（本地 valid 可用）
```

`--check` 会拒绝：表头错误、行数不符、`row_id` 跳号、`user_id`/`video_id` 与评测集不对齐、
`score` 非数字或为 NaN/Inf。**提交前请自行跑一遍 `--check`。**

## 从哪里开始改

下面的排序是**实测过的**，不是猜的。组委会已经试过的死路直接标出来，别重复踩。

### 已实测

| 试过的 | 结果 |
|---|---|
| **换损失函数为 BPR（pairwise）** | primary **0.5971** vs pointwise 的 **0.5946**（5 seed 均稳定，+0.0025 > 收敛阈值 ε=0.002）。**有收益，已设为默认**（`--loss pairwise`）。同时试了 listwise（组内 softmax，0.5931，比 pointwise 还低）和 LambdaRank@5（BPR × \|ΔnDCG@5\|，0.5891，两个子指标都更差 —— 78% 的采样 pair 因排在 top-5 之外被截断权重清零，梯度信号不足）。细节见 `RUN_LOG.md`。 |
| **用户历史序列（DIN 风格 attention）** —— `sequence_model.py`，PyTorch，最近 160 条 long_view=1 历史对候选 video_id 做 attention，接入 FM 交互项 | primary **0.5967**（第一版）/ **0.5969**（把 same_video 精确命中标志显式拼进 attention 输入后），都跟 BPR FM 的 **0.5971** 在噪声内无差别。**attention 机制本身两次都没有收益。** |
| **↳ 但序列信号本身是真实存在的** —— `prior_exposure`（是否精确复看过这个视频，二值，`ablation_prior_exposure.py`）+0.0015；`author_recency`（离上次看这个作者的作品过了多久，分桶，`ablation_author_recency.py`）+0.0017；`adjacency`（是否紧接在同一作者的一次 long_view 之后，二值，`ablation_adjacency.py`）单独也有 **0.5986**，几乎完全接住 author_recency 的收益——说明这个时序信号本质是 session 边界的阶梯效应，不是 DIEN 假设的平滑衰减。**两个都有收益，已收编进 `data.py`/`baseline.py` 默认（见上面 Baseline 阶梯，+0.0037）。** 结论：不是"没有序列信号可挖"，是 DIN 的 softmax attention 没能从这个数据集里学会利用它；这对要不要投入 DIEN 是个负面信号，BST（靠自注意力捕捉顺序，不假设平滑演化）不受影响，仍是待测方向。完整推导见 `RUN_LOG.md`。 |
| **多任务（is_click）** —— `baseline.py --loss pairwise_multitask`，BPR 主任务 + is_click 辅助 BCE（共享 embedding，各自独立一阶项），`aux_weight` 试了 0.2 和 1.0 | primary **0.6007**（weight=0.2，5 seed）/ **0.5999**（weight=1.0，单 seed），跟当前最优 7 域 FM 的 **0.6008** 在噪声内无差别，权重更大甚至略降。**没有收益，两个权重都试过，不是权重没调对。** 推测：is_click 和 long_view 相关但是漏斗的不同阶段，共享 embedding 可能已经被主任务的 BPR 梯度训练得足够好，辅助任务没有再挤出新信息，反而分走了一部分梯度预算。细节见 `RUN_LOG.md`。 |
| **多任务（CWM 风格观看时长删失回归）** —— `baseline.py --loss pairwise_watchtime`（默认），BPR 主任务 + 观看时长辅助回归。播完的行（17.3%）不能直接拿 play_time_ms 当目标——中位数只超出 1.09 倍，但有个从 p90 开始暴涨到 802 倍的极端拖尾（大概率是放着不管的循环播放，不是真兴趣），所以播完的行统一用 duration_ms 当单侧下界（削失回归/Tobit 损失），没播完的行才用 play_time_ms 精确回归 | primary **0.6017** vs 纯 BPR 7 域的 **0.6008**（5 seed 均稳定，+0.0009，标准误差算约 3.6σ，但比 BPR/时序特征那几步的效应量小）。**有收益，已设为默认。** 之所以比 is_click 有效：play_time_ms>=18s 单独就能猜中 96.7% 的 long_view，说明 long_view 本来就是观看时长离散化后的版本，不是一个独立信号——辅助任务教的是同一件事的更细粒度版本，不是另一件相关的事。细节见 `RUN_LOG.md`。 |
| **换模型（DeepFM）** —— `deepfm_model.py`，PyTorch，在 FM 交互项上并行加一个小 DNN 分支（7 域 embedding 拼接过 2 层 MLP，`z = z_FM + z_DNN`），跟纯 BPR FM 7 域（不叠 watchtime）对照 | primary **0.6007** vs **0.6008**，几乎完全无差别（−0.0001）。**没有收益。** 跟容量消融是同一个结论的又一次印证：`user_id × video_id` 交叉已经吃掉了这批特征里大部分可学信号，不管额外表达能力来自哪（更大 embedding / DNN 非线性组合）都挖不出新东西。 |
| **↳ 换模型（FinalMLP）** —— `finalmlp_model.py`，PyTorch，完全弃用显式交互结构：两路 MLP，各自对同一份 embedding 做独立的 MMOE 风格门控加权，再用多头双线性融合两路输出（AAAI 2023，在多个基准上超过 DCNv2/xDeepFM），跟 DeepFM 用同一对照基准 | primary **0.6002** vs **0.6008**（−0.0006，5 seed 稳定偏低，比 DeepFM 还差一点）。**没有收益，比"加法接入"的 DeepFM 还略差。** 推测：FinalMLP 完全没有显式双线性项，`user_id × video_id` 这个交叉只能靠两路 MLP 从零学出来，比 FM 精确解析这个交叉更难学，114 万行数据的规模下这个劣势没被它自己的架构优势抵消。DeepFM/FinalMLP 两次独立确认同一个结论——**换模型这条路线已经彻底摸清楚了**，问题不在架构表达能力，在这批特征本身没有更多可挖的结构。 |
| **加静态特征** —— 把 CWM 的 13 个特征域全接进来（+`music_id`/`video_type`/`upload_type` + 6 个用户侧粗桶） | primary **0.5940** vs 5 域的 **0.5950**，噪声内无差别，甚至略降。**没有收益。** |
| **加模型容量** —— embedding 维度 k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887，几乎不动。**没有收益。** |

原因：`user_id × video_id` 的交叉已经吃掉了大部分可学的信号。`follow_user_num_range` 这类粗桶
在 `user_id` 面前是冗余的；而 114 万行数据也撑不起更大的容量。**瓶颈不在特征和容量。**

⚠️ 另外注意：**纯用户侧特征的一阶项对分数贡献恒为 0。** 因为排序在用户内部做，任何在用户内为常数的项
都不改变组内顺序（实测：`item_pop × 用户偏置` 和纯 `item_pop` 的分数一位不差）。用户侧特征只能通过
**与物品侧的交叉项**起作用。

### 未探索：headroom 应该在这里

按我们判断的可能性排序（**这几条组委会没测过，是留给你们的**）：

1. **BST（自注意力捕捉顺序，不是 DIEN 的平滑演化假设）。** DIN attention 本身没有收益，但序列
   信号确实存在且已经被两个手工特征吃掉大半（见上）；DIEN 的核心机制（GRU 平滑演化）被
   `author_recency` 的阶梯形状证伪，但 BST 不假设平滑演化，仍是合理的下一步——如果还想在
   attention 这条路上继续投入的话。`sequence.py`/`sequence_model.py`（`--arch bst`）的历史构建 +
   torch 训练基础设施已经现成，但训练在本机 CPU 上比 DIN 慢约 35 倍（self-attention 对小
   head_dim 在 CPU 上效率很差），**正在等 GPU 集群跑**（`train_seq.sbatch`），结果见 `RUN_LOG.md`。
2. **时间特征与分布漂移。** `hourmin`、`date`，以及 train 与 test 之间的漂移。
3. **无偏验证（进阶）。** `log_random_4_22_to_5_08_pure.csv` 是随机曝光日志（118 万行），
   可作为额外的无偏验证集，检查模型是否只在有偏流量上过拟合。

## 用你自己的模型（包括 CWM）

`evaluate.py` 与模型完全解耦，它只要三个等长数组：

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores 可以来自任何模型
```

- `user_ids`：评测集每一行的 user_id
- `labels`：该行的 `long_view`（0/1）
- `scores`：你的模型给该行打的分（任意实数，只用相对大小）

所以你可以完全不用 `baseline.py`，换成 PyTorch、LightGBM 或 [CWM](https://github.com/hyz20/CWM) 的 xDeepFM，
只要最后把 `scores` 交给 `evaluate()` 即可。**评分口径由 `evaluate.py` 唯一决定。**

> 用 CWM 需注意：它依赖 `torch==1.6.0`（2020 年版本，新 GPU 上大概装不上），
> 且它的损失优化的是 counterfactual watch time、评测标签是自己重建的 `long_view2`。
> 它是一篇时长纠偏论文的研究代码，可以当**进阶参考**，不建议作为起步点。
>
> 本 kit 已经在 `baseline.py --loss pairwise_watchtime`（默认）里实现了 CWM 的核心思想
> （观看时长删失回归，见上面 Baseline 阶梯 + `RUN_LOG.md`）——不是完整复刻这篇论文，
> **评测目标依然是官方的 `long_view`，不是 CWM 自己重建的 `long_view2`**，watch-time 只是
> 辅助任务，不改变评分口径。

## 文件

| | |
|---|---|
| `evaluate.py` | 指标实现 + 全部口径约定。**不要改。** |
| `data.py` | 数据加载、官方划分、特征编码。默认 7 域（5 静态 + `prior_exposure` + `author_recency`）。加静态特征改 `_raw_fields`。`aux_labels()` 提供多任务辅助标签（如 `is_click`）；`watch_time_targets()` 提供 CWM 风格删失回归目标。 |
| `temporal_features.py` | `prior_exposure`/`author_recency` 两个跨行时序特征的构建逻辑（numpy-only），`data.py` 的 `encode()` 默认调用。 |
| `baseline.py` | 三个 baseline。FM 是要打败的那个。`--loss` 支持 pointwise/pairwise/listwise/lambdarank/pairwise_multitask/pairwise_watchtime（默认）/pairwise_combined（is_click+watchtime 一起练，无额外收益）。 |
| `baseline_scores.json` | 官方发布的分数 + 种子方差 + 收敛参数——**历史快照**（5 域 + pointwise），不随后续改动更新，跟当前默认配置的数字不再一致，当前数字见 README 的 Baseline 阶梯。 |
| `submit.py` | 生成 / 校验提交文件。 |
| `ablation_features.py` | 特征消融实验，可复现「加特征没有收益」那组数字。 |
| `ablation_prior_exposure.py` | 诊断：`prior_exposure` 单独接 5 域 FM 的收益（+0.0015）。已收编进 `data.py` 默认，这里保留作为单独归因的记录。 |
| `ablation_author_recency.py` | 诊断：`author_recency` 单独接 5 域 FM 的收益（+0.0017），形状是阶梯而非平滑衰减。已收编进 `data.py` 默认。 |
| `ablation_adjacency.py` | 诊断：更简单的"是否紧接同作者一次 long_view 之后"二值特征，单独测出 0.5986，跟 `author_recency` 的收益几乎完全重合——证实阶梯效应就是全部收益来源。未收编（跟 `author_recency` 冗余）。 |
| `ablation_author_watch_affinity.py` | 诊断：观看时长当特征用（历史参与"深度"而非"时机"），单独接 5 域 FM +0.0009，比 `author_recency` 弱。未收编。 |
| `sequence.py` | 用户历史序列构建（numpy-only）。给 `sequence_model.py` 用，也可单独复用。 |
| `sequence_model.py` | DIN 风格历史 attention + FM（PyTorch）。两版（纯 pooling / 显式拼 same_video 标志）都没有收益，尽管序列信号本身确实存在（见上三行诊断）——问题在 DIN 的机制，不在数据。见 `RUN_LOG.md`。 |
| `train_seq.sbatch` | 在 SoC GPU 集群上跑 `sequence_model.py` 的 Slurm 作业脚本（5 seed 并行，`--array=0-4`）。 |
| `deepfm_model.py` | DeepFM（PyTorch）：FM 交互项上并行加一个小 DNN 分支。没有收益（干净的空结果），见 `RUN_LOG.md`。 |
| `finalmlp_model.py` | FinalMLP（PyTorch）：完全弃用显式交互结构的两流 MLP + 门控 + 多头双线性融合。没有收益，比 DeepFM 还略差，见 `RUN_LOG.md`。 |
| `RUN_LOG.md` | 每次改动的完整实验记录（命令、5 seed 结果、正负向结论）。 |
