"""
KAB2 incremental generation — export new SGFs from DB and run katago batch_analysis.
Does NOT train. Use auto_train.py for the full pipeline.

Usage:
    uv run python scripts/generate_kab2_incremental.py
    uv run python scripts/generate_kab2_incremental.py --visits 1 --workers 2
    uv run python scripts/generate_kab2_incremental.py --upgrade-lite --visits 1
"""

from __future__ import annotations
import argparse, logging, shutil, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("kab2_gen")

DB_DSN = "host=localhost port=5432 dbname=gopredict user=gopredict password=gopredict_dev"
KAB2_CACHE = Path.home() / ".katarank" / "kab2_cache"


def _run_chunk(chunk_ids: list[str], db_dsn: str, cache_dir: str,
               katago_cfg: dict, max_visits: int, chunk_label: str) -> int:
    """Process one chunk in a worker process. Returns count of .npz generated."""
    import psycopg2
    from auto_train import export_sgfs_by_ids, generate_kab2, copy_new_npz

    cache = Path(cache_dir)
    work_dir = Path(tempfile.mkdtemp(prefix="kab2_chunk_"))
    sgf_dir = work_dir / "sgfs"
    kab2_dir = work_dir / "kab2"

    try:
        conn = psycopg2.connect(db_dsn)
        try:
            n_exported = export_sgfs_by_ids(conn, set(chunk_ids), sgf_dir)
        finally:
            conn.close()

        if n_exported == 0:
            return 0

        generate_kab2(sgf_dir, kab2_dir, katago_cfg, max_visits=max_visits)
        n_copied = copy_new_npz(kab2_dir, cache)
        return n_copied
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Incremental KAB2 generation")
    parser.add_argument("--db-dsn", default=DB_DSN)
    parser.add_argument("--cache-dir", type=Path, default=KAB2_CACHE)
    parser.add_argument("--batch-size", type=int, default=500,
                        help="SGFs per chunk (default 500)")
    parser.add_argument("--visits", type=int, default=1,
                        help="Visits per move (1 = feature extraction only, default 1)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel katago processes (default 1; 2 if GPU memory allows)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upgrade-lite", action="store_true",
                        help="Move lite-only .npz to _lite/ first, regenerate as full")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from auto_train import (
        get_cached_game_ids, get_analyzed_game_ids,
        rebuild_meta, assign_train_val_split, _load_katago_config, purge_lite_npz,
    )
    import psycopg2

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    katago_cfg = _load_katago_config()

    if args.upgrade_lite:
        n = purge_lite_npz(cache_dir, dry_run=args.dry_run)
        log.info("Moved %d lite .npz to _lite/ — will regenerate as full", n)

    conn = psycopg2.connect(args.db_dsn)
    try:
        cached_ids = get_cached_game_ids(cache_dir)
        all_ids = get_analyzed_game_ids(conn)
        new_ids = all_ids - cached_ids
    finally:
        conn.close()

    log.info("Cached: %d | Analyzed: %d | New: %d | visits=%d workers=%d",
             len(cached_ids), len(all_ids), len(new_ids), args.visits, args.workers)

    if not new_ids:
        log.info("Nothing to generate.")
        return

    if args.dry_run:
        log.info("[DRY RUN] Would generate KAB2 for %d games", len(new_ids))
        return

    new_list = sorted(new_ids)
    chunks = [new_list[i:i + args.batch_size]
              for i in range(0, len(new_list), args.batch_size)]

    total_generated = 0

    if args.workers <= 1:
        for ci, chunk in enumerate(chunks):
            label = f"Chunk {ci+1}/{len(chunks)}"
            log.info("=== %s: %d games ===", label, len(chunk))
            n = _run_chunk(chunk, args.db_dsn, str(cache_dir),
                           katago_cfg, args.visits, label)
            total_generated += n
            log.info("%s done: +%d (total: %d)", label, n, total_generated)
    else:
        log.info("Starting %d parallel workers", args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for ci, chunk in enumerate(chunks):
                label = f"Chunk {ci+1}/{len(chunks)}"
                f = pool.submit(_run_chunk, chunk, args.db_dsn, str(cache_dir),
                                katago_cfg, args.visits, label)
                futures[f] = label
            for f in as_completed(futures):
                label = futures[f]
                try:
                    n = f.result()
                    total_generated += n
                    log.info("%s done: +%d (total: %d)", label, n, total_generated)
                except Exception as e:
                    log.error("%s failed: %s", label, e)

    log.info("=== All done: generated %d new KAB2 files ===", total_generated)

    log.info("Rebuilding _meta.csv from all .npz files...")
    rebuild_meta(cache_dir)
    assign_train_val_split(cache_dir)

    final_count = len(list(cache_dir.glob("*.npz")))
    log.info("Cache now has %d games total.", final_count)


if __name__ == "__main__":
    main()
