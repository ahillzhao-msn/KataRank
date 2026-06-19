#!/usr/bin/env python3
"""训练前验证 — 检查验证集的段位分布是否合理。

策略（三段式）：
  - 全局样本 ≥ 20：严格分层，每个段位分 5-10% 到验证集
  - 全局样本 3~19：分 1 个到验证集（聚焦在尽量有覆盖面，不是统计意义）
  - 全局样本 < 3：全部用于训练（没有分割价值）

主流段位（2k-6d）必须通过分层验证。
两端段位（20k-3k、7d-9d）按上述三档处理，不强制验证集存在。

Usage:
    uv run python scripts/validate_split.py --meta data/kab2/_meta.csv
"""

import argparse
import csv
import sys
from collections import Counter

RANK_NAMES = [f'{i}k' if i < 20 else f'{i-19}d' for i in range(29)]

# Band 定义（按段位序号分组，每组 ≈6 个段位）
BAND_RANGES = [
    (0, 5, '14k-20k (尾段)'),     # 14k=6 … 20k=0
    (6, 11, '8k-13k (低段)'),      # 8k=12 … 13k=7
    (12, 17, '2k-7k (中段)'),      # 2k=18 … 7k=13
    (18, 23, '2d-6d (高段)'),      # 2d=21 … 6d=17
    (24, 28, '7d-9d (顶段)'),      # 7d=24 … 9d=28
]
MAIN_BANDS = {1, 2, 3}  # 只对主流段位严格分层


def rank_to_idx(rank_str: str) -> int | None:
    try:
        parts = rank_str.split('_')
        num = int(parts[-1][:-1])
        suffix = parts[-1][-1]
        if suffix == 'k':
            return 20 - num
        elif suffix == 'd':
            return 20 + num - 1
    except:
        return None


def band_for(idx: int) -> int:
    for b, (lo, hi, _) in enumerate(BAND_RANGES):
        if lo <= idx <= hi:
            return b
    return 4  # fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', required=True)
    parser.add_argument('--min-samples', type=int, default=3,
                        help='Min val samples per mainstream band')
    args = parser.parse_args()

    # Per-rank counts
    rank_val = Counter()
    rank_train = Counter()
    # Per-band aggregates
    band_val = [0] * 5
    band_train = [0] * 5

    with open(args.meta) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get('set', '?')
            for side in ['B_humanRank', 'W_humanRank']:
                idx = rank_to_idx(row.get(side, ''))
                if idx is None:
                    continue
                b = band_for(idx)
                if split == 'T':
                    rank_train[idx] += 1
                    band_train[b] += 1
                elif split == 'V':
                    rank_val[idx] += 1
                    band_val[b] += 1

    total_train = sum(band_train)
    total_val = sum(band_val)
    total = total_train + total_val

    print(f'Meta: {args.meta}')
    print(f'Total labeled: {total} ({total_train} train + {total_val} val)')
    print()

    # Per-rank detail
    print(f'{"段位":>6} {"全局":>5} {"训练":>6} {"验证":>6} {"策略":>12}')
    print('-' * 40)
    all_ok = True
    for i in range(29):
        t = rank_train.get(i, 0)
        v = rank_val.get(i, 0)
        total_r = t + v

        if total_r >= 20:
            strategy = '分层5-10%'
        elif total_r >= 3:
            strategy = '分1个'
        else:
            strategy = '全部训练'

        # Check: if global >= 20, expect at least 1 in val
        if total_r >= 20 and v == 0:
            flag = ' ⚠️ 缺验证集'
            if band_for(i) in MAIN_BANDS:
                flag += ' ❌'
                all_ok = False
        elif total_r >= 3 and v == 0:
            flag = ' (允许空)'
        else:
            flag = ''

        print(f'{RANK_NAMES[i]:>6} {total_r:>5} {t:>6} {v:>6} {strategy:>12}{flag}')

    # Band-level summary for mainstream
    print()
    print('--- Band 汇总 ---')
    print(f'{"Band":>5} {"范围":>18} {"训练":>6} {"验证":>6} {"占比":>6}')
    print('-' * 45)
    for i in range(5):
        v = band_val[i]
        t = band_train[i]
        pct = v / (v + t) * 100 if (v + t) > 0 else 0

        if i in MAIN_BANDS:
            ok = v >= args.min_samples
            status = '✅' if ok else '❌'
            extra = f' (需≥{args.min_samples})' if not ok else ''
            if not ok:
                all_ok = False
        else:
            status = 'ℹ️'
            extra = ''

        print(f'{i:>5} {BAND_RANGES[i][2]:>18} {t:>6} {v:>6} {pct:>5.1f}%  {status}{extra}')

    print()
    if all_ok:
        print(f'✅ 通过：主流段位（Band 1-3）均 ≥ {args.min_samples} 个验证样本')
        sys.exit(0)
    else:
        print('❌ 不通过：主流段位验证集不足')
        sys.exit(1)


if __name__ == '__main__':
    main()
