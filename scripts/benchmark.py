#!/usr/bin/env python
"""
KataRank — Pipeline Benchmark (no model, pure data pipeline)
==============================================================
Usage:
    uv run python scripts/benchmark.py --sgf-dir <path> [--games 50]

Generates KAB2 data in 4 modes from SGF files at different rank levels,
then compares output consistency.

4 groups:
  1. File + full   (--no-trunk not set)  → expects 29-dim rank info
  2. File + lite   (--no-trunk)          → scalar only
  3. Stream + full                       → 29-dim rank info via pipe
  4. Stream + lite                       → scalar only via pipe
"""

import argparse, csv, glob, hashlib, json, os, re, subprocess, sys, tempfile
import time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def find_sgfs(sgf_dir, n_per_group=50):
    """Collect N SGFs per rank-level subdirectory."""
    sgf_dir = Path(sgf_dir)
    groups = sorted([d for d in sgf_dir.iterdir() if d.is_dir() and d.name[0].isdigit() or d.name == 'pro'])
    if not groups:
        groups = [sgf_dir]  # flat directory fallback
    all_sgfs = []
    for g in groups:
        sgfs = sorted(g.glob('*.sgf'))[:n_per_group]
        for s in sgfs:
            all_sgfs.append((str(s), g.name))
        if sgfs:
            print(f'  {g.name}: {len(sgfs)} SGFs')
    return all_sgfs


def run_batch(katago, model, human_model, sgfs, output_dir, mode='file', no_trunk=False):
    """Run katago batch_analysis, return (success_count, timing, stderr_text)."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, '_input.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['File', 'Player Black', 'Player White', 'Score',
                     'BlackRating', 'WhiteRating', 'Set'])
        for sgf_path, rank_group in sgfs:
            w.writerow([sgf_path, rank_group, f'{rank_group}_opp', 0.5, 1500, 1500, 'T'])

    cmd = [katago, 'batch_analysis', '-model', model,
           '-list', csv_path, '-output-dir', output_dir,
           '-min-moves', '10', '-profile']
    if human_model:
        cmd += ['-human-model', human_model]
    if no_trunk:
        cmd += ['-no-trunk']
    if mode == 'stream':
        cmd += ['-stream']

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    dt = time.time() - t0

    stderr = proc.stderr.decode()
    # Parse ok count
    ok = 0
    for l in stderr.split('\n'):
        m = re.search(r'ok=(\d+)', l)
        if m: ok = int(m.group(1))

    npz_files = sorted(glob.glob(os.path.join(output_dir, '*.npz')))
    return ok, dt, stderr, npz_files


def analyze_npz(npz_path, label=''):
    """Read a combined .npz file and return stats dict."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from katarank.data.preprocess import read_kab2_combined

    b_moves, w_moves, b_info, w_info = read_kab2_combined(npz_path)

    stats = {
        'file': os.path.basename(npz_path),
        'label': label,
        'b_num_moves': b_info['num_moves'],
        'w_num_moves': w_info['num_moves'],
        'input_dim': b_info['input_dim'],
        'b_mean_log_prior': round(float(b_info['mean_log_prior']), 4),
        'w_mean_log_prior': round(float(w_info['mean_log_prior']), 4),
        'b_human_rank': int(b_info.get('human_rank_idx', -1)),
        'w_human_rank': int(w_info.get('human_rank_idx', -1)),
        'b_human_lp': round(float(b_info.get('human_log_prior', 0)), 4),
        'w_human_lp': round(float(w_info.get('human_log_prior', 0)), 4),
    }
    # Hash of first 100 floats for consistency check
    flat = b_moves.flatten()[:100].tobytes() + w_moves.flatten()[:100].tobytes()
    stats['content_hash'] = hashlib.md5(flat).hexdigest()[:12]
    return stats


def main():
    parser = argparse.ArgumentParser(description='KataRank Pipeline Benchmark')
    parser.add_argument('--sgf-dir',
                        default=os.environ.get('KATARANK_SGF_CORPUS', os.path.expanduser('~/sgf-corpus')))
    parser.add_argument('--katago',
                        default=os.environ.get('KATAGO_BIN', 'katago'))
    parser.add_argument('--model',
                        default=os.environ.get('KATAGO_MODEL', os.path.expanduser('~/.katago/default_model.bin.gz')))
    parser.add_argument('--human-model', default=None)
    parser.add_argument('--games', type=int, default=50,
                        help='SGFs per rank level')
    parser.add_argument('--report', default=None,
                        help='Save JSON report')
    args = parser.parse_args()

    if not args.human_model:
        for p in [os.environ.get('KATAGO_HUMAN_MODEL', ''),
                  os.path.expanduser('~/.katago/b18c384nbt-humanv0.bin.gz')]:
            if os.path.exists(p):
                args.human_model = p
                break
        if not args.human_model:
            print('No human model found — rank info may show -1')

    print(f'KataRank Pipeline Benchmark')
    print(f'=' * 60)
    print(f'SGF dir:        {args.sgf_dir}')
    print(f'Games/group:    {args.games}')
    print(f'Human model:    {args.human_model or "(none)"}')

    # 1. Collect SGFs
    print(f'\n─── Step 1: Collecting SGFs ───')
    all_sgfs = find_sgfs(args.sgf_dir, args.games)
    if not all_sgfs:
        print('ERROR: no SGFs found!')
        return 1
    print(f'  Total: {len(all_sgfs)} SGFs')

    base_dir = os.path.join(tempfile.gettempdir(), '_katarank_pipe_bench')
    shutil.rmtree(base_dir, ignore_errors=True)

    # 2. Run 4 modes
    modes = [
        ('file_full',   False, False),
        ('file_lite',   False, True),
        ('stream_full', True,  False),
        ('stream_lite', True,  True),
    ]

    results = {}
    print(f'\n─── Step 2: Data Generation (4 modes) ───')

    for label, use_stream, no_trunk in modes:
        out_dir = os.path.join(base_dir, label)
        print(f'\n  [{label}] ', end='', flush=True)
        ok, dt, stderr, npz_files = run_batch(
            args.katago, args.model, args.human_model,
            all_sgfs, out_dir, mode='stream' if use_stream else 'file',
            no_trunk=no_trunk,
        )
        print(f'ok={ok}  time={dt:.1f}s  ({ok/max(0.01,dt):.1f} g/s)')

        # Parse profile summary
        profile = {}
        for l in stderr.split('\n'):
            if '[profile] aggregate' in l:
                continue
            m = re.search(r'total=(\d+).*ok=(\d+)', l)
            if m: pass

        # Analyze up to 6 npz files (one per rank group)
        sample_stats = []
        for npz_path in npz_files[:6]:
            stats = analyze_npz(npz_path, label)
            sample_stats.append(stats)

        results[label] = {
            'games': ok,
            'time_s': round(dt, 1),
            'games_per_s': round(ok / max(0.01, dt), 2),
            'npz_files': len(npz_files),
            'samples': sample_stats,
        }

    # 3. Cross-mode comparison
    print(f'\n─── Step 3: Consistency Check ───')

    # Compare file_full vs stream_full content hashes
    ff_samples = results.get('file_full', {}).get('samples', [])
    sf_samples = results.get('stream_full', {}).get('samples', [])
    fl_samples = results.get('file_lite', {}).get('samples', [])
    sl_samples = results.get('stream_lite', {}).get('samples', [])

    # Same game → same hash check (first N games)
    for i in range(min(len(ff_samples), len(sf_samples))):
        ff = ff_samples[i]
        sf = sf_samples[i]
        match = ff['content_hash'] == sf['content_hash']
        lp_match = ff['b_mean_log_prior'] == sf['b_mean_log_prior']
        print(f'  game {i}: file_full vs stream_full  hash={ff["content_hash"]}/{sf["content_hash"]} '
              f'match={match}  lp_match={lp_match}')

    # dim check
    print(f'  Full input_dim: {ff_samples[0]["input_dim"] if ff_samples else "N/A"}')
    print(f'  Lite input_dim: {fl_samples[0]["input_dim"] if fl_samples else "N/A"}')

    # Rank distribution check
    print(f'\n─── Step 4: 29-dim Rank Distribution ───')
    rank_counts = {}
    for label, samples in [('file_full', ff_samples), ('stream_full', sf_samples)]:
        for s in samples:
            for side in ['b', 'w']:
                r = s.get(f'{side}_human_rank', -1)
                if r >= 0:
                    key = f'{label}_{r}'
                    rank_counts[key] = rank_counts.get(key, 0) + 1

    # Show rank distribution per mode
    for label in ['file_full', 'stream_full', 'file_lite', 'stream_lite']:
        samples = results.get(label, {}).get('samples', [])
        if not samples:
            continue
        ranks = [s[f'{side}_human_rank'] for s in samples for side in ['b','w']
                 if s.get(f'{side}_human_rank', -1) >= 0]
        if ranks:
            from collections import Counter
            dist = Counter(ranks)
            top = dist.most_common(5)
            print(f'  {label:15s}: top ranks = {top}')

    # 4. Report
    print(f'\n─── Summary ───')
    for label in ['file_full', 'file_lite', 'stream_full', 'stream_lite']:
        r = results.get(label, {})
        print(f'  {label:15s}: {r.get("games",0)} games  {r.get("time_s",0)}s  '
              f'{r.get("games_per_s",0)} g/s  dim={r.get("samples",[{}])[0].get("input_dim","?")}')

    # Cleanup
    shutil.rmtree(base_dir, ignore_errors=True)

    report = {'results': results}
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nReport: {args.report}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
