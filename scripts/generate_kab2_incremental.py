"""
KAB2 incremental generation — export new SGFs from DB and run katago batch_analysis.
Does NOT train. Use auto_train.py for the full pipeline.

Usage:
    uv run python scripts/generate_kab2_incremental.py
    uv run python scripts/generate_kab2_incremental.py --batch-size 500  # process in chunks
"""

from __future__ import annotations
import argparse, logging, shutil, tempfile
from pathlib import Path

log = logging.getLogger("kab2_gen")

DB_DSN = "host=localhost port=5432 dbname=gopredict user=gopredict password=gopredict_dev"
KAB2_CACHE = Path.home() / ".katarank" / "kab2_cache"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Incremental KAB2 generation")
    parser.add_argument("--db-dsn", default=DB_DSN)
    parser.add_argument("--cache-dir", type=Path, default=KAB2_CACHE)
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Process SGFs in chunks of this size")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upgrade-lite", action="store_true",
                        help="Delete lite-only .npz files first so they get regenerated as full")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from auto_train import (
        get_cached_game_ids, get_analyzed_game_ids, export_sgfs_by_ids,
        generate_kab2, copy_new_npz, merge_meta, rebuild_meta,
        assign_train_val_split, _load_katago_config, purge_lite_npz,
    )
    import psycopg2

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    katago_cfg = _load_katago_config()

    if args.upgrade_lite:
        n = purge_lite_npz(cache_dir, dry_run=args.dry_run)
        log.info("Purged %d lite .npz files — they will be regenerated as full", n)

    conn = psycopg2.connect(args.db_dsn)
    try:
        cached_ids = get_cached_game_ids(cache_dir)
        all_ids = get_analyzed_game_ids(conn)
        new_ids = all_ids - cached_ids

        log.info("Cached: %d | Analyzed: %d | New: %d", len(cached_ids), len(all_ids), len(new_ids))

        if not new_ids:
            log.info("Nothing to generate.")
            return

        if args.dry_run:
            log.info("[DRY RUN] Would generate KAB2 for %d games", len(new_ids))
            return

        new_list = sorted(new_ids)
        total_generated = 0

        for i in range(0, len(new_list), args.batch_size):
            chunk = set(new_list[i : i + args.batch_size])
            chunk_num = i // args.batch_size + 1
            total_chunks = (len(new_list) + args.batch_size - 1) // args.batch_size
            log.info("=== Chunk %d/%d: %d games ===", chunk_num, total_chunks, len(chunk))

            work_dir = Path(tempfile.mkdtemp(prefix="kab2_chunk_"))
            sgf_dir = work_dir / "sgfs"
            kab2_dir = work_dir / "kab2"

            try:
                n_exported = export_sgfs_by_ids(conn, chunk, sgf_dir)
                log.info("Exported %d SGFs", n_exported)

                if n_exported == 0:
                    continue

                generate_kab2(sgf_dir, kab2_dir, katago_cfg)
                n_copied = copy_new_npz(kab2_dir, cache_dir)
                merge_meta(cache_dir, kab2_dir)
                total_generated += n_copied
                log.info("Chunk done: +%d npz (total new: %d)", n_copied, total_generated)
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

        log.info("=== All done: generated %d new KAB2 files ===", total_generated)

        log.info("Rebuilding _meta.csv from all .npz files...")
        rebuild_meta(cache_dir)
        assign_train_val_split(cache_dir)

        final_count = len(list(cache_dir.glob("*.npz")))
        log.info("Cache now has %d games total.", final_count)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
