# KuaiRand-Pure Starter Kit

## ⚠️ 先看这个：哪些文件属于被评分的任务

赛题口径（**已按完整题面核对过**，见 `RUN_LOG.md` 2026-08-30 的口径核对记录）：
**用户内排序** —— 对每个用户在评测集里的曝光重排，正例是 `long_view`，
指标 **GAUC / nDCG@5**，primary = 两者平均。三个数据集变体（Pure / 1k / 27k）
口径完全一致。

| 属于评分任务 | 说明 |
|---|---|
| `evaluate.py` | 官方评分代码，**不要改** |
| `data.py` / `temporal_features.py` | 数据加载与特征 |
| `baseline.py` | FM + 各种 loss |
| `make_submission.py` | **产出最终提交文件**（按 validation-best 选） |
| `sequence_model.py` / `deepfm_model.py` / `finalmlp_model.py` / `lightgcn_model.py` | 试过的各种模型 |
| `ablation_*.py` | 各项消融 |
| `run_bonus.py` / `bonus_fm_torch.py` / `data_large.py` | bonus 数据集（1k / 27k），同一套任务和指标。`run_bonus.py`=numpy FM（1K 可用，27K 因稠密 Adam 更新不现实）；`bonus_fm_torch.py`=torch SparseFM（稀疏 embedding 更新，27K 用这个，需要 GPU） |

| ❌ **不属于**评分任务 | 说明 |
|---|---|
| `evaluate_retrieval.py` | 全库检索的 NDCG@10 / Recall@50 |
| `retrieval_baseline.py` | 检索版 random / pop / BPR-MF |
| `retrieval_lightgcn.py` | 检索版 LightGCN |

这三个文件是一次**走错方向**的产物：赛题 Constraints 表的 "Limits" 行写着
`NDCG@10 / Recall@50, click = positive`，与题面其余 8 处（含评分代码、提交 schema、
官方 baseline 数字）矛盾。决定性证据是提交格式为"每个评测行一个分数"，
而全库检索根本无法用这个 schema 表达。文件保留但**不参与评分**，
完整分析见 `RUN_LOG.md`。

## 资源消耗（赛题 Feasibility 项要求）

| 项目 | 数值 |
|---|---|
| 迭代/实验条目 | `RUN_LOG.md` 21 条记录，覆盖约 35 组配置（每条常含多个 5-seed sweep） |
| 主要基准 wall-clock | 单次 FM 训练约 30-40 秒（单核 CPU）；5-seed sweep 约 3-5 分钟 |
| 最重的单项 | BST 在 CPU 上约 2450 秒/epoch → 改用 GPU 集群后约 22 秒/epoch（约 110 倍） |
| Bonus 1K | 单次运行 1874 秒（31 分钟），峰值内存 2.68 GB（本机 CPU） |
| Bonus 27K | 单次运行 4038 秒（67 分钟），GPU（TITAN V），真实峰值内存约 41GB（SparseFM，稀疏 embedding 更新） |
| GPU 用量 | BST 5 个 seed（约 15 分钟）+ 27K SparseFM 一次（约 67 分钟），SoC 集群；其余全部单核 CPU |
| 硬件限制 | 本机 8 GB RAM 挡住了 27K 用 numpy FM 的稠密更新（dim≈4090 万时纯粹是浪费计算量，不只是内存问题）和 BST 的本地重训；集群 GPU 节点（125GB RAM）+ 稀疏 embedding 实现解决了两者 |

**人工干预次数.** 本次运行**不是全自动**。按性质分两类：
- **任务指派**（"接下来试 X"）：约 15 次，属于正常的人类设定方向
- **纠正 / 解阻**：约 10 次，其中影响结果的关键几次：
  1. 放开 torch 依赖（我原本误以为 numpy-only 是硬约束）
  2. 提供 GPU 集群访问，并协助排查 SSH / VPN / 磁盘配额（约 6 轮往返）
  3. **对我"集成已到头"的结论提出质疑**（"I believe there is more here"）
     —— 直接导致了本项目最好的结果（异构集成 0.6034）
  4. **提供完整题面，纠正了我错误的任务口径转向**

第 3 和第 4 条特别值得记录：一次是我过早下了否定结论，一次是我拿单行文字
推翻了多处一致证据。两次都是人类干预纠正的，**不是 agent 自查发现的**。

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

test 集上的分数。**单模型要打败的是加粗那一行（当前默认配置，0.6017）；最后一行是集成，目前的最好成绩（0.6034）。**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random（下界，自检用） | 0.4996 | 0.4511 | 0.4753 |
| item popularity（trivial） | 0.6308 | 0.5121 | 0.5715 |
| FM + pointwise，5 域（原官方 baseline，`baseline_scores.json` 里的历史快照） | 0.6610 | 0.5282 | 0.5946 |
| FM + BPR，5 域 | 0.6638 | 0.5304 | 0.5971 |
| FM + BPR，7 域（+ `prior_exposure` + `author_recency`） | 0.6689 | 0.5326 | 0.6008 |
| **FM + BPR + CWM 风格观看时长辅助任务，7 域（当前默认单模型）** | **0.6702** | **0.5333** | **0.6017** |
| 异构集成（BST + 2 个 FM 变体，组内 z-score 融合，`ablation_hetero_ensemble.py`） | 0.6724 | 0.5344 | **0.6034** |

前三步提升都超过收敛阈值 ε=0.002，5 个 seed 上稳定复现：损失函数从 pointwise 换成
BPR（+0.0025，另外试过 listwise/LambdaRank@5，均更差）、再加两个时序特征（+0.0037，
组合收益略高于两者单独收益之和）。最后一步（+0.0009）幅度更小，但 5 个 seed 稳定，
用标准误差算显著性约 3.6σ（`RUN_LOG.md` 有完整讨论）；同样试过 `is_click` 辅助任务，
无收益。完整演变过程和每一步的具体实验设计见 `RUN_LOG.md`。

**最后一行（0.6034）是目前最好的数字**，但它是集成而非单模型：把 BST 和两个 FM 变体的
预测融合起来，比最好的单模型高 +0.0017（约 4σ）。**表里前面几行保持单模型口径**——
`RUN_LOG.md` 里约 30 条实验记录都是以单模型 0.6017 为对照基准的，混用两种口径会让那些
对比难以解读。要冲分就用集成，要跟历史实验对比就用单模型。

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
| **↳ 但序列信号本身是真实存在的** —— `prior_exposure`（是否精确复看过这个视频，二值，`ablation_prior_exposure.py`）+0.0015；`author_recency`（离上次看这个作者的作品过了多久，分桶，`ablation_author_recency.py`）+0.0017；`adjacency`（是否紧接在同一作者的一次 long_view 之后，二值，`ablation_adjacency.py`）单独也有 **0.5986**，几乎完全接住 author_recency 的收益——说明这个时序信号本质是 session 边界的阶梯效应，不是 DIEN 假设的平滑衰减。**两个都有收益，已收编进 `data.py`/`baseline.py` 默认（见上面 Baseline 阶梯，+0.0037）。** 结论：不是"没有序列信号可挖"，是 DIN 的 softmax attention 没能从这个数据集里学会利用它。 |
| **↳ BST（GPU 上验证）** —— `sequence_model.py --arch bst`，SoC 集群 GPU（~22s/epoch，比 CPU 快约 110 倍），自注意力+位置编码替换 DIN 的无序 pooling | primary **0.6014**（5 seed），比 DIN 高 **+0.0045~0.0047**（10~12 个标准误差，这份记录里最不含糊的一个结果）——**证实了顺序信息确实有用，DIN 的问题就是机制本身**。但跟当前 7 域 FM 干净基线（0.6008）比只高 +0.0006（约 1.5 个标准误差，没到平时的确认门槛），跟当前最优（含 watchtime，0.6017）比几乎打平（−0.0003）。**没有超过现有最优，不采用**——self-attention 挖到的信号跟手工时序特征 + watchtime 辅助任务已经挖到的高度重合。完整推导（含集群踩坑记录）见 `RUN_LOG.md`。 |
| **多任务（is_click）** —— `baseline.py --loss pairwise_multitask`，BPR 主任务 + is_click 辅助 BCE（共享 embedding，各自独立一阶项），`aux_weight` 试了 0.2 和 1.0 | primary **0.6007**（weight=0.2，5 seed）/ **0.5999**（weight=1.0，单 seed），跟当前最优 7 域 FM 的 **0.6008** 在噪声内无差别，权重更大甚至略降。**没有收益，两个权重都试过，不是权重没调对。** 推测：is_click 和 long_view 相关但是漏斗的不同阶段，共享 embedding 可能已经被主任务的 BPR 梯度训练得足够好，辅助任务没有再挤出新信息，反而分走了一部分梯度预算。细节见 `RUN_LOG.md`。 |
| **多任务（CWM 风格观看时长删失回归）** —— `baseline.py --loss pairwise_watchtime`（默认），BPR 主任务 + 观看时长辅助回归。播完的行（17.3%）不能直接拿 play_time_ms 当目标——中位数只超出 1.09 倍，但有个从 p90 开始暴涨到 802 倍的极端拖尾（大概率是放着不管的循环播放，不是真兴趣），所以播完的行统一用 duration_ms 当单侧下界（削失回归/Tobit 损失），没播完的行才用 play_time_ms 精确回归 | primary **0.6017** vs 纯 BPR 7 域的 **0.6008**（5 seed 均稳定，+0.0009，标准误差算约 3.6σ，但比 BPR/时序特征那几步的效应量小）。**有收益，已设为默认。** 之所以比 is_click 有效：play_time_ms>=18s 单独就能猜中 96.7% 的 long_view，说明 long_view 本来就是观看时长离散化后的版本，不是一个独立信号——辅助任务教的是同一件事的更细粒度版本，不是另一件相关的事。细节见 `RUN_LOG.md`。 |
| **换模型（DeepFM）** —— `deepfm_model.py`，PyTorch，在 FM 交互项上并行加一个小 DNN 分支（7 域 embedding 拼接过 2 层 MLP，`z = z_FM + z_DNN`），跟纯 BPR FM 7 域（不叠 watchtime）对照 | primary **0.6007** vs **0.6008**，几乎完全无差别（−0.0001）。**没有收益。** 跟容量消融是同一个结论的又一次印证：`user_id × video_id` 交叉已经吃掉了这批特征里大部分可学信号，不管额外表达能力来自哪（更大 embedding / DNN 非线性组合）都挖不出新东西。 |
| **↳ 换模型（FinalMLP）** —— `finalmlp_model.py`，PyTorch，完全弃用显式交互结构：两路 MLP，各自对同一份 embedding 做独立的 MMOE 风格门控加权，再用多头双线性融合两路输出（AAAI 2023，在多个基准上超过 DCNv2/xDeepFM），跟 DeepFM 用同一对照基准 | primary **0.6002** vs **0.6008**（−0.0006，5 seed 稳定偏低，比 DeepFM 还差一点）。**没有收益，比"加法接入"的 DeepFM 还略差。** 推测：FinalMLP 完全没有显式双线性项，`user_id × video_id` 这个交叉只能靠两路 MLP 从零学出来，比 FM 精确解析这个交叉更难学，114 万行数据的规模下这个劣势没被它自己的架构优势抵消。DeepFM/FinalMLP 两次独立确认同一个结论——**换模型这条路线已经彻底摸清楚了**，问题不在架构表达能力，在这批特征本身没有更多可挖的结构。 |
| **Dynamic Negative Sampling** —— `baseline.py --loss pairwise_dns`，每个正例从 `dns_n` 个候选负例里挑当前模型打分最高的一个（而非随机），理论上跟 Top-K 指标关联更强 | `dns_n=8` **训练不稳定**（loss 不降反升，GAUC/nDCG@5 一起往下掉，不是 LambdaRank 那种指标之间此消彼长）——热身 3 epoch + 切换时 lr×0.2 两个标准修复手段都没能挽救，只是让崩溃变慢。`dns_n=2`（更温和）训练稳定，但结果是干净的空转：primary **0.6006** vs **0.6008**（−0.0002，噪声内）。**没有收益。** 推测：这个数据集的物品池只有 7,538 个、复看率极高，"模型当前打分最高的负例"很可能是用户其实感兴趣、只是这次没看够阈值的视频——难负例挖掘理论上跟大物品池场景（负例基本可靠）不一样，这里挖到的更像是标签噪声而非真信号。细节见 `RUN_LOG.md`。 |
| **训练集时间加权 / 截断** —— `ablation_train_window.py`。先发现一个此前没记录的数据结构事实：**训练集前 4 天（Apr 9-12）占了全部训练数据的 64%，曝光强度 7.4 impressions/user/day，而 valid/test 只有约 1.1**——同一批用户（91% 的 test 用户在 early train 出现过），是记录强度变了，不是人群变了 | 硬截断（只保留跟评测同强度的尾部）和软加权（保留全部数据、按时间指数衰减采样）**两种都单调变差**：硬截断 0.6020→0.5978→0.5928→0.5795（保留比例 100%→36%→17%→7.5%）；软加权半衰期 3/7/14 天分别是 0.5932/0.5996/0.5999，最温和的那档都够不着均匀采样的 0.6020。**没有收益。** 结论：**数据量压倒分布匹配**——FM 的参数以 `user_id`/`video_id` embedding 为主，每个 ID 需要足够的交互量才估得准，为了分布对齐而饿着它们是笔亏本买卖。那个 7 倍强度差本身是这个数据集值得知道的性质，但不构成可用的建模杠杆。细节见 `RUN_LOG.md`。 |
| **观看时长辅助任务的分位数重参数化（RAD, AAAI 2025）** —— `baseline.py --wt_target quantile`，不回归观看时长绝对值，改回归它在**同时长组**经验分布里的分位数（天然抗离群，不需要手工截断）。先验证了前提确实成立：时长分组**内部**的 corr(观看时长, long_view) 是 0.46~0.64，高于全局的 0.596，说明时长确实混淆了信号 | 目标质量提升非常明显——跟 long_view 的相关系数从 **0.596 涨到 0.825**——但 primary **0.6016** vs 现默认 log 目标的 **0.6017**，完全打平（−0.0001）。**没有收益。** 这是本项目最尖锐的一个"目标变好但指标不动"的例子：辅助任务原本那 +0.0009 的收益来自"给共享 embedding 喂同一个信号的更细粒度版本"（`play_time>=18s` 单独就能猜中 96.7% 的 long_view），一旦 embedding 吸收完这部分，再怎么打磨辅助目标的参数化都没有增量——天花板由 `user_id × video_id` 能表达什么决定，不由训练信号的保真度决定。细节见 `RUN_LOG.md`。 |
| **自适应去噪训练（ADT, WSDM 2021）** —— `baseline.py --loss pairwise_adt`，按 loss 大小给 pair 降权（`w=sigmoid(zpos-zneg)^beta`），噪声交互在训练早期表现为大 loss。**这是 DNS 那条路的反向验证**：DNS 专挑高 loss 样本（训崩了），ADT 反过来给它们降权 | beta=0.25/0.5/1.0/2.0（降权 pair 占比 0.8%/8.8%/30%/39%）对应 primary **0.6013/0.6007/0.6003/0.5979**——降权越狠越差，任何强度都够不着 0.6017。**没有收益。** 但跟 DNS 合起来结论更强：**高 loss 样本携带的是真信号，不是可丢弃的噪声**（两个方向都试过了：加权训崩、降权变差）。这也**部分推翻了 DNS 那条记录里"难负例基本是标签噪声"的推测**——那个假设能解释 DNS 崩掉，但它预测 ADT 应该有用，而实测没有。细节见 `RUN_LOG.md`。 |
| **种子集成（同构）** —— `ablation_ensemble.py`，把 N 个 seed 训出的模型的**预测分**平均 | 单模型均值 0.6018 → 集成 **0.6025**（+0.0007，约 1.8σ，3 个模型即饱和）。**有小幅收益但很快见顶**——成员之间只差随机种子，误差高度相关，能削减的方差有限。见下一行。 |
| **异构集成** —— `ablation_hetero_ensemble.py`，把**架构不同**的 5 个模型（BST / DeepFM / FinalMLP / 2 个 FM 变体）的预测融合；试了 raw / z-score / rank / RRF 四种融合方式；**融合方式和成员子集只在 valid 上选，test 只报一次** | **0.6034**（valid 选出 z-score + {bst, fm_quantile, fm_watchtime}），比最好的单模型高 **+0.0017（约 4σ）**、比同构集成高 **+0.0009（约 2.3σ）**。**这是目前项目最好的结果。** 机制在成员相关性矩阵里：**BST 跟其它成员的 rank 相关只有 0.885~0.892，而非 BST 成员两两之间是 0.926~0.973**（DeepFM 和 FinalMLP 甚至高达 0.973——架构看着天差地别，其实在算同一件事）。差异在**读取的信息**（BST 用了顺序）才带来互补误差；差异在**怎么算交互**则不会。这也部分推翻了上一行的结论（"方差削减动不了信息天花板"——对同构集成成立，但加入 BST 是引入新信息，属于降低偏差）。见 `RUN_LOG.md`。 |
| **加静态特征** —— 把 CWM 的 13 个特征域全接进来（+`music_id`/`video_type`/`upload_type` + 6 个用户侧粗桶） | primary **0.5940** vs 5 域的 **0.5950**，噪声内无差别，甚至略降。**没有收益。** |
| **加模型容量** —— embedding 维度 k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887，几乎不动。**没有收益。** |

原因：`user_id × video_id` 的交叉已经吃掉了大部分可学的信号。`follow_user_num_range` 这类粗桶
在 `user_id` 面前是冗余的；而 114 万行数据也撑不起更大的容量。**瓶颈不在特征和容量。**

⚠️ 另外注意：**纯用户侧特征的一阶项对分数贡献恒为 0。** 因为排序在用户内部做，任何在用户内为常数的项
都不改变组内顺序（实测：`item_pop × 用户偏置` 和纯 `item_pop` 的分数一位不差）。用户侧特征只能通过
**与物品侧的交叉项**起作用。

### 未探索：headroom 应该在这里

按我们判断的可能性排序（**这几条组委会没测过，是留给你们的**）：

1. **无偏验证（进阶）。** `log_random_4_22_to_5_08_pure.csv` 是随机曝光日志（118 万行），
   可作为额外的无偏验证集，检查模型是否只在有偏流量上过拟合。注意这是**诊断工具，不是
   提分杠杆**——主分永远在有偏的标准日志上算，这条不会直接抬高分数，但对理解模型行为
   （以及写报告）有价值，也是这个数据集本身的研究主线（KuaiRand, CIKM 2022）。
2. **`hourmin` 时段特征。** 分布漂移那一面已经试过了（见上，没有收益），但一天内的时段
   本身还没当特征接进来过。考虑到静态特征加了一轮又一轮都是空结果，期望值不高。

> ⚠️ **先读这个再挑方向。** 损失函数、静态特征、模型容量、模型架构（DeepFM / FinalMLP）、
> 序列建模（DIN / BST）、多任务、负采样策略、去噪训练、训练集时间加权、辅助目标重参数化
> ——**这些方向单独试都收敛到同一个天花板（单模型 0.601~0.602）**：这批特征上
> `user_id × video_id` 交叉之外几乎没有可挖的结构，跟机制是否先进无关。真正有效的几步
> （BPR 换损失 +0.0025、时序特征 +0.0037、观看时长辅助任务 +0.0009）都是**便宜的、
> 针对性的**改动，不是更复杂的模型。
>
> **但"单独试"这三个字很重要。** 异构集成把其中几个"各自失败"的模型合起来，拿到了目前
> 最好的 0.6034——关键成员正是 BST，它单独看只是打平，但它跟 FM 系的预测相关性明显更低
> （0.885~0.892 vs 非 BST 成员之间的 0.926~0.973），**错在不同的地方**。教训是：
> 一个模型单独打不赢，不代表它没价值；**判断一个方向该不该丢，光看它自己的分数不够。**
>
> 所以如果要继续往上推：与其再试一个"更强的架构"（DeepFM 和 FinalMLP 的预测相关性高达
> 0.973，说明换算法不换信息源，本质是同一个模型），不如找**读取了不同信息的**模型——
> 就像 BST 用了顺序那样。

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
| `baseline.py` | 三个 baseline。FM 是要打败的那个。`--loss` 支持 pointwise/pairwise/listwise/lambdarank/pairwise_multitask/pairwise_watchtime（默认）/pairwise_combined（is_click+watchtime 一起练，无额外收益）/pairwise_dns（hard negative mining，无收益，`dns_n` 大了还会训练不稳定）/pairwise_adt（按 loss 降权去噪，无收益，降得越狠越差）。`--wt_target` 可选 log（默认）/ quantile（RAD 风格分位数目标，实测与 log 打平）。 |
| `baseline_scores.json` | 官方发布的分数 + 种子方差 + 收敛参数——**历史快照**（5 域 + pointwise），不随后续改动更新，跟当前默认配置的数字不再一致，当前数字见 README 的 Baseline 阶梯。 |
| `submit.py` | 生成 / 校验提交文件。 |
| `ablation_features.py` | 特征消融实验，可复现「加特征没有收益」那组数字。 |
| `ablation_prior_exposure.py` | 诊断：`prior_exposure` 单独接 5 域 FM 的收益（+0.0015）。已收编进 `data.py` 默认，这里保留作为单独归因的记录。 |
| `ablation_author_recency.py` | 诊断：`author_recency` 单独接 5 域 FM 的收益（+0.0017），形状是阶梯而非平滑衰减。已收编进 `data.py` 默认。 |
| `ablation_adjacency.py` | 诊断：更简单的"是否紧接同作者一次 long_view 之后"二值特征，单独测出 0.5986，跟 `author_recency` 的收益几乎完全重合——证实阶梯效应就是全部收益来源。未收编（跟 `author_recency` 冗余）。 |
| `ablation_author_watch_affinity.py` | 诊断：观看时长当特征用（历史参与"深度"而非"时机"），单独接 5 域 FM +0.0009，比 `author_recency` 弱。未收编。 |
| `ablation_train_window.py` | 诊断：训练集按时间硬截断 / 软加权，测"训练分布该不该向评测期对齐"。两种都单调变差——数据量压倒分布匹配。见 `RUN_LOG.md`。 |
| `ablation_ensemble.py` | 同构集成（成员只差 seed）：+0.0007，3 个模型即饱和。见 `RUN_LOG.md`。 |
| `ablation_hetero_ensemble.py` | **异构集成：项目最好结果 0.6034。** 分成员训练并缓存到 `scores/*.npz`（`--member <name>`）再融合（`--combine`）——一次性训 5 个模型会爆内存。见 `RUN_LOG.md`。 |
| `sequence.py` | 用户历史序列构建（numpy-only）。给 `sequence_model.py` 用，也可单独复用。 |
| `sequence_model.py` | `--arch din`：两版都没有收益，问题在机制不在数据。`--arch bst`：GPU 上验证，比 DIN 高 10+ 个标准误差（证实顺序信息有用），但跟当前最优打平，不采用。见 `RUN_LOG.md`。 |
| `train_seq.sbatch` | 在 SoC GPU 集群上跑 `sequence_model.py` 的 Slurm 作业脚本（5 seed 并行，`--array=0-4`）。 |
| `deepfm_model.py` | DeepFM（PyTorch）：FM 交互项上并行加一个小 DNN 分支。没有收益（干净的空结果），见 `RUN_LOG.md`。 |
| `finalmlp_model.py` | FinalMLP（PyTorch）：完全弃用显式交互结构的两流 MLP + 门控 + 多头双线性融合。没有收益，比 DeepFM 还略差，见 `RUN_LOG.md`。 |
| `RUN_LOG.md` | 每次改动的完整实验记录（命令、5 seed 结果、正负向结论）。 |
