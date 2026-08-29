"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np
from temporal_features import build_temporal_features

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个静态类别域 + 2 个跨行时序域（prior_exposure / author_recency，见
# temporal_features.py，已在 RUN_LOG.md 验证过有稳定收益，2026-08-28 收编为默认特征）。
# 想加静态域就往 _raw_fields 加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'prior_exposure', 'author_recency']

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             int(r['time_ms']), 1 if r['is_click'] != '0' else 0,
                             float(r['play_time_ms'])))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def _raw_fields(x, edges):
    return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

def build_vocab(splits):
    """只用 train 建词表，只管 5 个静态类别域（不含 prior_exposure/author_recency
    这两个跨行时序域——它们不需要 vocab，直接是小整数类别，见 encode()）。
    未见过的取值统一落到该域的 UNK 槽。
    返回 (vocabs, unk, field_dims, offsets, edges) —— encode()、历史序列构建共用。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])
    n_static = len(_raw_fields(tr[0], edges))       # 5，与 FIELDS 的长度（7）无关，避免多算
    vocabs = [dict() for _ in range(n_static)]
    for x in tr:
        for i, v in enumerate(_raw_fields(x, edges)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    return vocabs, unk, field_dims, offsets, edges

def encode(splits):
    """把类别特征映射成连续 id：5 个静态域走 vocab 查表，另外 2 个跨行时序域
    （prior_exposure/author_recency）直接是小整数类别，接在后面，共用同一张 embedding
    表的地址空间。未见过的静态取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 dim。"""
    vocabs, unk, field_dims, offsets, edges = build_vocab(splits)
    raw = lambda x: _raw_fields(x, edges)
    temporal = build_temporal_features(splits)      # {split: (N,2) int32}
    dim5 = int(sum(field_dims))
    temporal_dims = [2, 11]                          # prior_exposure ∈{0,1}；author_recency ∈{0..10}
    t_off = np.cumsum([dim5] + temporal_dims[:-1]).astype(np.int32)
    dim = dim5 + sum(temporal_dims)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        X[:, 5] = temporal[name][:, 0] + t_off[0]
        X[:, 6] = temporal[name][:, 1] + t_off[1]
        enc[name] = (X, y, users)
    return enc, int(dim)

def aux_labels(splits, col='is_click'):
    """辅助任务标签（多任务用，见 baseline.py 的 loss='pairwise_multitask'）。
    跟 encode() 用同一个行序（都是直接 iterate splits[name]），所以
    aux_labels(splits)[name] 和 encode(splits)[0][name] 逐行对齐。"""
    idx = {'is_click': 8}[col]
    return {name: np.array([x[idx] for x in rws], dtype=np.float32)
            for name, rws in splits.items()}

def watch_time_targets(splits, scale=12.0):
    """CWM 风格的观看时长删失回归目标（见 baseline.py 的 loss='pairwise_watchtime'）。
    censored=True（play_time_ms >= duration_ms，播完了）的行只知道"真实观看时长
    至少是 duration_ms"这个下界，不是精确值——用 duration_ms（不是 play_time_ms
    本身，后者在放循环的极端情况下可以到 duration 的几百倍，是噪声，不是信号）
    作为单侧 loss 的阈值 tau；没播完的行是精确观测，直接回归 t=log1p(play_time_ms)。
    scale 只是把 log1p(ms) 量级（~9-13）压到接近 1，方便 aux_weight 跟其它任务
    可比——不是什么理论上有意义的常数。
    返回 {split: (t, tau, censored)}，t/tau 为 float32 (N,)，censored 为 bool (N,)。"""
    out = {}
    for name, rws in splits.items():
        pt = np.array([x[9] for x in rws], dtype=np.float64)
        dur = np.array([x[5] for x in rws], dtype=np.float64)
        censored = pt >= dur
        t = (np.log1p(pt) / scale).astype(np.float32)
        tau = (np.log1p(dur) / scale).astype(np.float32)
        out[name] = (t, tau, censored)
    return out
