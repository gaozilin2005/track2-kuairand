"""KuaiRand-1K / 27K 的省内存加载器（本机只有 8GB RAM，data.py 那套撑不住）。

为什么需要另写一个：`data.py` 的 `load()` 把每一行做成一个 Python tuple 存进 list。
Pure 有 140 万行，大约 280MB，没问题；但 1K 有 1170 万行（约 2.3GB），27K 有 3.22 亿行
（约 64GB）——光是 Python 对象开销就装不下，跟模型无关，纯粹是数据结构的问题。

这里改成**列式**：一边流式读 CSV 一边把类别值映射成整数，只保留 numpy 数组，
从头到尾不建 Python tuple 列表。同样的 1170 万行大约只要 400-500MB。

口径跟 data.py 完全一致（这点很重要，否则跟 Pure 的结果不可比）：
  - 同样的日期划分 train 20220408-0421 / valid 0422-0428 / test 0429-0508
  - 同样 5 个静态域 user_id / video_id / author_id / tab / dur_bucket
  - 同样只用 train 建词表，未见过的取值落 UNK 槽
  - dur_bucket 同样按 train 的 duration 分 10 个分位桶
时序特征（prior_exposure / author_recency）这里暂不构建——它们要按 (user, author) 分组
做全量时间扫描，在这个数据量下是另一个量级的工程，先把 baseline 打通。
"""
import csv, glob, os
import numpy as np

SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']


def _code(d, key):
    v = d.get(key)
    if v is None:
        v = len(d); d[key] = v
    return v


def load_columnar(data_dir, suffix='1k', verbose=True):
    """流式读日志，返回 {split: dict of numpy columns}。
    列：user, video, author, tab, dur（原始毫秒，之后再分桶）, label(long_view), click。"""
    vid2author = {}
    with open(os.path.join(data_dir, f'video_features_basic_{suffix}.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
    if verbose:
        print(f'  video features: {len(vid2author)} videos')

    u_d, v_d, a_d, t_d = {}, {}, {}, {}
    cols = {name: {k: [] for k in ('user', 'video', 'author', 'tab', 'dur', 'label', 'click')}
            for name in SPLITS}

    # 27K 把每个日期段的日志切成了 part1/part2（Pure/1K 都是单文件）——用 glob 通配，
    # 排序后按 part1→part2 顺序读，不需要在磁盘上拼接（省掉重复存一份 ~9GB 数据，
    # 也避免拼接时中间嵌进第二份文件自己的表头行导致那一行解析出错）。
    n_read = 0
    for pattern in (f'log_standard_4_08_to_4_21_{suffix}*.csv',
                    f'log_standard_4_22_to_5_08_{suffix}*.csv'):
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not paths:
            raise SystemExit(f'no files matching {pattern} in {data_dir}')
        if verbose and len(paths) > 1:
            print(f'  {pattern} -> {len(paths)} parts: {[os.path.basename(p) for p in paths]}')
        for path in paths:
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    d = int(r['date'])
                    split = None
                    for name, (lo, hi) in SPLITS.items():
                        if lo <= d <= hi:
                            split = name; break
                    if split is None:
                        continue
                    c = cols[split]
                    c['user'].append(_code(u_d, r['user_id']))
                    c['video'].append(_code(v_d, r['video_id']))
                    c['author'].append(_code(a_d, vid2author.get(r['video_id'], 'UNK')))
                    c['tab'].append(_code(t_d, r['tab']))
                    c['dur'].append(float(r['duration_ms']))
                    c['label'].append(1 if r['long_view'] != '0' else 0)
                    c['click'].append(1 if r['is_click'] != '0' else 0)
                    n_read += 1
                    if verbose and n_read % 2_000_000 == 0:
                        print(f'    {n_read/1e6:.0f}M rows ...')

    out = {}
    for name, c in cols.items():
        out[name] = {
            'user': np.asarray(c['user'], dtype=np.int32),
            'video': np.asarray(c['video'], dtype=np.int32),
            'author': np.asarray(c['author'], dtype=np.int32),
            'tab': np.asarray(c['tab'], dtype=np.int32),
            'dur': np.asarray(c['dur'], dtype=np.float32),
            'label': np.asarray(c['label'], dtype=np.float32),
            'click': np.asarray(c['click'], dtype=np.int8),
        }
        c.clear()
    if verbose:
        print('  rows per split:', {k: len(v['user']) for k, v in out.items()})
        print(f'  vocab: {len(u_d)} users, {len(v_d)} videos, {len(a_d)} authors, {len(t_d)} tabs')
    return out, (len(u_d), len(v_d), len(a_d), len(t_d))


def encode_columnar(data, vocab_sizes, n_dur_buckets=10, verbose=True):
    """把列式数据编码成 FM 用的 X (N,5) int32，各域拼进同一个 embedding 地址空间。
    词表大小按**全量**取（流式读时已经统一编码），每域末尾留一个 UNK 槽保持跟
    data.py 结构一致——虽然这里不会真的用到（编码时所有取值都已见过）。"""
    n_u, n_v, n_a, n_t = vocab_sizes
    edges = np.quantile(data['train']['dur'], np.linspace(0, 1, n_dur_buckets + 1)[1:-1])
    dims = [n_u + 1, n_v + 1, n_a + 1, n_t + 1, n_dur_buckets + 1]
    offs = np.cumsum([0] + dims[:-1]).astype(np.int64)
    dim = int(sum(dims))

    enc = {}
    for name, c in data.items():
        n = len(c['user'])
        X = np.empty((n, 5), dtype=np.int32)
        X[:, 0] = c['user'] + offs[0]
        X[:, 1] = c['video'] + offs[1]
        X[:, 2] = c['author'] + offs[2]
        X[:, 3] = c['tab'] + offs[3]
        X[:, 4] = np.searchsorted(edges, c['dur']).astype(np.int32) + offs[4]
        enc[name] = (X, c['label'], c['user'])
    if verbose:
        print(f'  encoded dim={dim}, X memory per split: ' +
              ', '.join(f'{k}={v[0].nbytes/1e6:.0f}MB' for k, v in enc.items()))
    return enc, dim
