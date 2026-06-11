"""
KataRank — Inference CLI
=========================
Rank Go players from SGF records. The REST API (katarank.api.server) is a
thin shell over the same workflow functions.

Usage::

    # Files / directory / stdin (any combination)
    katarank-infer game1.sgf game2.sgf --model kata1.bin.gz
    katarank-infer --sgf-dir ./games/ --model kata1.bin.gz
    cat game.sgf | katarank-infer --stdin --model kata1.bin.gz

    # With a trained KataRankModel (full 29-class rank distribution)
    katarank-infer game.sgf --model kata1.bin.gz --checkpoint nets/best.pt

    # Archive results (extension picks the format)
    katarank-infer --sgf-dir ./games/ --model kata1.bin.gz \\
        --output results.jsonl.gz

Without --checkpoint the output falls back to raw engine statistics
(meanLogPrior as rating; HumanSL rank if -human-model was given).
Output default: one KAB2Output JSON object per line on stdout.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from katarank.engine import KataGoEngine
from katarank.schema import save_outputs_batch, rank_idx_to_str
from katarank.workflow import run_rank_files, run_rank_strings


def main():
    parser = argparse.ArgumentParser(
        prog='katarank-infer',
        description='Go player rank assessment from SGF records',
    )
    parser.add_argument('sgf_paths', nargs='*', help='SGF file paths')
    parser.add_argument('--sgf-dir',     default=None, help='Directory scanned for *.sgf')
    parser.add_argument('--stdin',       action='store_true',
                        help='Read one SGF content string from stdin')
    parser.add_argument('--model',       required=True, help='KataGo model .bin.gz')
    parser.add_argument('--checkpoint',  default=None,
                        help='KataRankModel .pt (omit for raw engine statistics)')
    parser.add_argument('--katago-bin',  default=None, help='KataGo binary path')
    parser.add_argument('--config',      default=None, help='KataGo config .cfg')
    parser.add_argument('--human-model', default=None,
                        help='HumanSL model .bin.gz (optional for inference)')
    parser.add_argument('--mode',        default='lite', choices=['lite', 'full'])
    parser.add_argument('--min-moves',   type=int, default=10)
    parser.add_argument('--device',      default=None, help='cpu / cuda')
    parser.add_argument('--output',      default=None,
                        help='Write results to file (.json/.jsonl/.json.gz/.jsonl.gz); '
                             'default: JSON lines on stdout')
    args = parser.parse_args()

    # ── Resolve inputs ────────────────────────────────────────────────────────
    paths: List[str] = list(args.sgf_paths)
    if args.sgf_dir:
        d = Path(args.sgf_dir)
        if not d.is_dir():
            sys.exit(f"ERROR: --sgf-dir not found: {args.sgf_dir}")
        paths += sorted(str(p) for p in d.glob('*.sgf'))
    strings: List[str] = []
    if args.stdin:
        strings.append(sys.stdin.read())

    if not paths and not strings:
        sys.exit("ERROR: no input (give SGF paths, --sgf-dir, or --stdin)")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        sys.exit(f"ERROR: SGF not found: {missing[:5]}")

    # ── Engine + optional model ───────────────────────────────────────────────
    engine = KataGoEngine(
        model       = args.model,
        config      = args.config,
        human_model = args.human_model,
        katago_bin  = args.katago_bin,
    )

    inf_workflow = None
    if args.checkpoint:
        from katarank.model import KataRankModel
        from katarank.workflow import InferenceWorkflow
        rank_model = KataRankModel.load(args.checkpoint)
        inf_workflow = InferenceWorkflow(rank_model, engine, device=args.device)

    # ── Run ───────────────────────────────────────────────────────────────────
    results = []
    if paths:
        results += run_rank_files(engine, inf_workflow, paths,
                                  mode=args.mode, min_moves=args.min_moves)
    if strings:
        results += run_rank_strings(engine, inf_workflow, strings,
                                    mode=args.mode, min_moves=args.min_moves)

    # ── Emit ──────────────────────────────────────────────────────────────────
    if args.output:
        save_outputs_batch(results, args.output)
        print(f"{len(results)} games -> {args.output}", file=sys.stderr)
    else:
        for r in results:
            print(json.dumps(dict(r), ensure_ascii=False))

    # Human-readable summary on stderr
    for r in results:
        print(
            f"  {r['game_id']}: B={rank_idx_to_str(r['b_rank'])} "
            f"({r['b_rating']:+.3f})  W={rank_idx_to_str(r['w_rank'])} "
            f"({r['w_rating']:+.3f})",
            file=sys.stderr,
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
