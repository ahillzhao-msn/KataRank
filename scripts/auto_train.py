"""
KataRank — Automatic Training Trigger

Connects to GoPredict PostgreSQL, checks if enough new analyzed games
have accumulated since the last training run, and if so:
  1. Exports NEW SGFs (only games without cached KAB2)
  2. Runs KataGo batch_analysis on new SGFs only (incremental)
  3. Merges new + cached _meta.csv, assigns T/V split
  4. Trains the KataRank model on the full dataset
  5. Records the training run in GoPredict's model_versions table

KAB2 .npz files are cached at ~/.katarank/kab2_cache/ so subsequent
runs only need to generate features for newly analyzed games.

Usage:
    uv run python scripts/auto_train.py              # check & train if threshold met
    uv run python scripts/auto_train.py --dry-run     # just report, don't train
    uv run python scripts/auto_train.py --force        # train regardless of threshold
    uv run python scripts/auto_train.py --threshold 5000
    uv run python scripts/auto_train.py --rebuild-cache  # regenerate all KAB2 from scratch
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

log = logging.getLogger("auto_train")

# ── Defaults ─────────────────────────────────────────────────────────────────

TRAIN_THRESHOLD = 3000
STATE_FILE = Path.home() / ".katarank" / "train_state.json"
KAB2_CACHE = Path.home() / ".katarank" / "kab2_cache"

DB_DSN = "host=localhost port=5432 dbname=gopredict user=gopredict password=gopredict_dev"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KATARANK_TRAIN = [sys.executable, "-m", "katarank.train.training"]

_TOML_PATH = Path.home() / ".katarank" / "server.toml"


def _load_katago_config() -> dict:
    cfg = {}
    if not _TOML_PATH.exists():
        return cfg
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            for line in _TOML_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#") and not line.startswith("["):
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"')
            return cfg
    with open(_TOML_PATH, "rb") as f:
        data = tomllib.load(f)
    return data.get("katarank", {})


# ── State persistence ────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"last_trained_count": 0, "runs": []}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


# ── KAB2 cache ──────────────────────────────────────────────────────────────

def get_cached_game_ids(cache_dir: Path) -> set[str]:
    """Return game IDs that already have .npz in the cache."""
    if not cache_dir.exists():
        return set()
    return {p.stem for p in cache_dir.glob("*.npz")}


def load_cached_meta(cache_dir: Path) -> list[dict]:
    """Load _meta.csv from cache, returns empty list if absent."""
    meta = cache_dir / "_meta.csv"
    if not meta.exists():
        return []
    with meta.open(encoding="utf-8", errors="replace") as f:
        return list(csvmod.DictReader(f))


def merge_meta(cache_dir: Path, new_dir: Path) -> int:
    """Merge new _meta.csv rows into cache _meta.csv. Returns total row count."""
    cached_rows = load_cached_meta(cache_dir)
    cached_ids = {r["file"] for r in cached_rows}

    new_meta = new_dir / "_meta.csv"
    new_rows = []
    if new_meta.exists():
        with new_meta.open(encoding="utf-8", errors="replace") as f:
            for row in csvmod.DictReader(f):
                if row["file"] not in cached_ids:
                    new_rows.append(row)

    if not new_rows and not cached_rows:
        return 0

    all_rows = cached_rows + new_rows
    fieldnames = list(all_rows[0].keys())
    meta_out = cache_dir / "_meta.csv"
    with meta_out.open("w", encoding="utf-8", newline="") as f:
        writer = csvmod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    log.info("Merged meta: %d cached + %d new = %d total",
             len(cached_rows), len(new_rows), len(all_rows))
    return len(all_rows)


def copy_new_npz(new_dir: Path, cache_dir: Path) -> int:
    """Copy .npz files from new_dir to cache_dir. Returns count copied."""
    n = 0
    for src in new_dir.glob("*.npz"):
        dst = cache_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            n += 1
    return n


# ── DB queries ───────────────────────────────────────────────────────────────

def count_analyzed_games(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT count(DISTINCT g.id)
            FROM games g
            JOIN game_ratings gr ON gr.game_id = g.id
        """)
        return cur.fetchone()[0]


def get_analyzed_game_ids(conn) -> set[str]:
    """Get all analyzed game IDs (UUID strings)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT g.id::text
            FROM games g
            JOIN game_ratings gr ON gr.game_id = g.id
            WHERE g.sgf_content IS NOT NULL
              AND length(g.sgf_content) > 50
        """)
        return {row[0] for row in cur}


def export_sgfs_by_ids(conn, game_ids: set[str], output_dir: Path) -> int:
    """Export SGFs for specific game IDs. Returns count written."""
    if not game_ids:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_list = list(game_ids)
    n = 0
    # Batch query in chunks to avoid oversized IN clauses
    chunk_size = 500
    for i in range(0, len(ids_list), chunk_size):
        chunk = ids_list[i : i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id::text, sgf_content FROM games
                WHERE id::text IN ({placeholders})
            """, chunk)
            for game_id, sgf in cur:
                (output_dir / f"{game_id}.sgf").write_text(sgf, encoding="utf-8")
                n += 1
    return n


# ── KAB2 generation ─────────────────────────────────────────────────────────

def generate_kab2(sgf_dir: Path, output_dir: Path, katago_cfg: dict) -> int:
    """Run KataGoEngine.batch_to_files() to produce .npz + _meta.csv."""
    from katarank.engine import KataGoEngine

    model = katago_cfg.get("model")
    human_model = katago_cfg.get("human_model")
    katago_bin = katago_cfg.get("katago_bin")

    if not model:
        from katarank.katago_setup import discover_model
        model = discover_model(None)

    engine = KataGoEngine(
        model=model,
        human_model=human_model,
        katago_bin=katago_bin,
    )

    sgf_paths = sorted(sgf_dir.glob("*.sgf"))
    log.info("Generating KAB2 for %d SGFs → %s", len(sgf_paths), output_dir)

    rc = engine.batch_to_files(
        output_dir=str(output_dir),
        sgf_paths=[str(p) for p in sgf_paths],
        mode="full",
    )
    if rc != 0:
        raise RuntimeError(f"katago batch_analysis failed (exit code {rc})")

    npz_count = len(list(output_dir.glob("*.npz")))
    meta = output_dir / "_meta.csv"
    log.info("Generated %d .npz files, _meta.csv exists: %s", npz_count, meta.exists())
    return npz_count


def assign_train_val_split(cache_dir: Path, val_fraction: float = 0.1):
    """Assign T/V split to _meta.csv. Preserves existing V assignments."""
    meta = cache_dir / "_meta.csv"
    if not meta.exists():
        raise FileNotFoundError(f"No _meta.csv in {cache_dir}")

    with meta.open(encoding="utf-8", errors="replace") as f:
        rows = list(csvmod.DictReader(f))
    if not rows:
        raise ValueError("Empty _meta.csv")

    # Count existing V assignments
    existing_val = {r["file"] for r in rows if r.get("set") == "V"}
    n_target_val = max(1, int(len(rows) * val_fraction))

    if len(existing_val) >= n_target_val:
        log.info("T/V split already adequate: %d val / %d total", len(existing_val), len(rows))
        return

    # Need more val games — pick from rows currently marked T or unmarked
    candidates = [r for r in rows if r["file"] not in existing_val]
    random.seed(42)
    random.shuffle(candidates)
    need = n_target_val - len(existing_val)
    new_val = {r["file"] for r in candidates[:need]}

    for row in rows:
        if row["file"] in existing_val or row["file"] in new_val:
            row["set"] = "V"
        else:
            row["set"] = "T"

    fieldnames = list(rows[0].keys())
    with meta.open("w", encoding="utf-8", newline="") as f:
        writer = csvmod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_val = len(existing_val) + len(new_val)
    log.info("Assigned split: %d train, %d val (%d preserved, %d new)",
             len(rows) - n_val, n_val, len(existing_val), len(new_val))


# ── Training ─────────────────────────────────────────────────────────────────

def run_training(kab2_dir: Path, resume_from: str | None = None) -> Path:
    """Run katarank-train and return the checkpoint path."""
    ckpt_dir = PROJECT_ROOT / "nets" / "katarank"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = KATARANK_TRAIN + [
        "--data-dir", str(kab2_dir),
    ]
    if resume_from and Path(resume_from).exists():
        cmd += ["--resume", resume_from]

    log.info("Running training: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {result.returncode})")

    best = ckpt_dir / "best.pt"
    if not best.exists():
        candidates = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            best = candidates[-1]
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    return best


# ── Record to GoPredict DB ──────────────────────────────────────────────────

def record_training_run(conn, ckpt_path: Path, num_games: int, metrics: dict | None = None):
    """Insert a row into model_versions in GoPredict's DB."""
    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO model_versions
                (version, artifact_path, num_training_games, metrics, active, created_at)
            VALUES (%s, %s, %s, %s, true, now())
        """, (
            version,
            str(ckpt_path),
            num_games,
            json.dumps(metrics or {}),
        ))
        cur.execute("""
            UPDATE model_versions SET active = false
            WHERE version != %s
        """, (version,))
    conn.commit()
    log.info("Recorded model version %s (%d games)", version, num_games)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="KataRank auto-training trigger")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check threshold and report, don't train")
    parser.add_argument("--force", action="store_true",
                        help="Train regardless of game count threshold")
    parser.add_argument("--threshold", type=int, default=TRAIN_THRESHOLD,
                        help=f"Trigger training every N new games (default {TRAIN_THRESHOLD})")
    parser.add_argument("--db-dsn", default=DB_DSN,
                        help="PostgreSQL connection string")
    parser.add_argument("--resume", default=None,
                        help="Resume training from this checkpoint")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Wipe KAB2 cache and regenerate all from scratch")
    parser.add_argument("--cache-dir", type=Path, default=KAB2_CACHE,
                        help=f"KAB2 cache directory (default {KAB2_CACHE})")
    args = parser.parse_args()

    state = _load_state()
    katago_cfg = _load_katago_config()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_cache:
        log.info("Rebuilding cache: wiping %s", cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("Connecting to GoPredict DB...")
    conn = psycopg2.connect(args.db_dsn)

    try:
        total = count_analyzed_games(conn)
        last = state.get("last_trained_count", 0)
        delta = total - last
        next_threshold = last + args.threshold
        cached_ids = get_cached_game_ids(cache_dir)

        log.info("Analyzed games: %d | Last trained at: %d | Delta: %d | Next trigger: %d",
                 total, last, delta, next_threshold)
        log.info("KAB2 cache: %d games at %s", len(cached_ids), cache_dir)

        if not args.force and total < next_threshold:
            log.info("Below threshold (%d < %d). No training needed.", total, next_threshold)
            return

        # ── Identify new games needing KAB2 ─────────────────────────────────
        all_analyzed_ids = get_analyzed_game_ids(conn)
        new_ids = all_analyzed_ids - cached_ids

        log.info("Games: %d analyzed, %d cached, %d new to generate",
                 len(all_analyzed_ids), len(cached_ids), len(new_ids))

        if args.dry_run:
            log.info("[DRY RUN] Would generate KAB2 for %d new games, "
                     "then train on %d total", len(new_ids), len(all_analyzed_ids))
            return

        # ── Step 1: Export only NEW SGFs ────────────────────────────────────
        if new_ids:
            log.info("=== Step 1/5: Exporting %d new SGFs ===", len(new_ids))
            work_dir = Path(tempfile.mkdtemp(prefix="katarank_incr_"))
            sgf_dir = work_dir / "sgfs"
            new_kab2_dir = work_dir / "kab2"

            n_exported = export_sgfs_by_ids(conn, new_ids, sgf_dir)
            log.info("Exported %d new SGFs", n_exported)

            if n_exported == 0:
                log.warning("No SGFs exported — skipping KAB2 generation")
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                # ── Step 2: Generate KAB2 for new games only ────────────────
                log.info("=== Step 2/5: Generating KAB2 for %d new games ===", n_exported)
                generate_kab2(sgf_dir, new_kab2_dir, katago_cfg)

                # ── Step 3: Merge into cache ────────────────────────────────
                log.info("=== Step 3/5: Merging into cache ===")
                n_copied = copy_new_npz(new_kab2_dir, cache_dir)
                merge_meta(cache_dir, new_kab2_dir)
                log.info("Copied %d new .npz to cache", n_copied)

                shutil.rmtree(work_dir, ignore_errors=True)
        else:
            log.info("All games already cached — skipping KAB2 generation")

        # ── Step 4: Assign T/V split on full cache ──────────────────────────
        log.info("=== Step 4/5: Preparing T/V split ===")
        assign_train_val_split(cache_dir)

        # ── Step 5: Train ───────────────────────────────────────────────────
        log.info("=== Step 5/5: Training KataRank model ===")
        resume = args.resume
        if not resume:
            candidate = PROJECT_ROOT / "nets" / "katarank" / "best.pt"
            if candidate.exists():
                resume = str(candidate)
        ckpt = run_training(cache_dir, resume_from=resume)
        log.info("Training complete → %s", ckpt)

        # ── Record ──────────────────────────────────────────────────────────
        n_total = len(get_cached_game_ids(cache_dir))
        record_training_run(conn, ckpt, n_total)

        state["last_trained_count"] = total
        state["runs"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "games": total,
            "cached_kab2": n_total,
            "new_generated": len(new_ids),
            "checkpoint": str(ckpt),
        })
        _save_state(state)
        log.info("Done. Next training trigger at %d analyzed games.", total + args.threshold)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
