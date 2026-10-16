"""
KataRank — Gated training session (docs/../GOPredict/docs/TRAINING_FLYWHEEL.md §3).

Runs the same incremental pipeline as auto_train.py (steps 1-4: export
new SGFs, generate KAB2 for them only, merge into cache, rebuild
_meta.csv + T/V split) but trains into a STAGING checkpoint dir instead
of overwriting the live production checkpoint directly, then applies an
evaluation gate before promoting:

    1-4. (identical to auto_train.py, reused from it directly)
    5. Train into nets/_candidate/ (resumed from the live checkpoint)
    6. Evaluate BOTH the live checkpoint and the candidate on the same
       fresh holdout split (apples-to-apples — historical
       training_report.json metrics may come from an older, smaller
       holdout)
    7. Candidate strictly better (lower val_loss) → promote: copy into
       the live checkpoint path, record model_versions (active=true,
       deactivating the previous row), and have the running server
       reload it (auto_train.reload_models — verified by sha256; result
       field "served")
       Candidate not better → discard: record model_versions with
       active=false (audit trail — a training run happened and did not
       improve on production), live checkpoint untouched

This is the ONE difference from auto_train.py's own default behavior,
which promotes unconditionally — that's what TRAINING_FLYWHEEL.md's
"评估闸门是法" (the evaluation gate is the law) rule exists to prevent.

Progress is written to ~/.katarank/train_job.json after every phase so
api/server.py's /train/status endpoint (and GoPredict's proxy of it) can
poll without parsing log output. Usage:

    uv run python scripts/train_session.py
    uv run python scripts/train_session.py --force --threshold 500
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_train as at  # noqa: E402  (reuse its steps 1-4 + state/DB helpers)

log = logging.getLogger("train_session")

JOB_FILE = Path.home() / ".katarank" / "train_job.json"
LIVE_CKPT = at.PROJECT_ROOT / "nets" / "katarank" / "best.pt"
CANDIDATE_DIR = at.PROJECT_ROOT / "nets" / "_candidate"
ARCHIVE_DIR = at.PROJECT_ROOT / "nets" / "katarank_archive"
KATARANK_URL_DEFAULT = "http://127.0.0.1:8765"

# config_kata_native.yaml's patience=30 budget is sized for training a
# model from nothing. A continuation run (--resume, with optimizer_state.pt
# restored — see training.py's Trainer) starts from an already-converged
# model with only the newly-accumulated games added, so genuine improvement
# should show within a handful of epochs; spending the full from-scratch
# patience budget waiting it out just burns hours for no reason.
#
# epochs is deliberately left at the config default (150), NOT shortened:
# the LR scheduler's internal state (last_epoch, T_max=epochs-warmup) is
# restored from optimizer_state.pt too, so a shorter epochs here would
# rebuild the cosine schedule with a different T_max than the one that
# state was checkpointed under — same last_epoch counter, different curve
# shape, producing a nonsensical LR. Keeping epochs constant across every
# resumed session makes the schedule one continuous 150-epoch curve spread
# across many short sessions; only patience controls how quickly any given
# session gives up and exits.
INCREMENTAL_PATIENCE = 5


# ── Job status file (polled by /train/status) ───────────────────────────────

def _write_job(**fields) -> None:
    JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if JOB_FILE.exists():
        try:
            state = json.loads(JOB_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    state.update(fields)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    JOB_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _phase(name: str, **extra) -> None:
    log.info("=== %s ===", name)
    _write_job(phase=name, running=True, error=None, **extra)


# ── Evaluation gate ──────────────────────────────────────────────────────────

def _training_loss_weights() -> dict:
    """Load the same loss_weights training.py trains with, so the gate's
    val_loss is computed under the exact weighting the checkpoints were
    actually optimized against — not KataRankLoss()'s own unrelated
    constructor defaults (see evaluate.py's docstring)."""
    import yaml
    config_path = at.PROJECT_ROOT / "src" / "katarank" / "train" / "config_kata_native.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("training", {}).get("loss_weights", {})


def _evaluate(ckpt_path: Path, cache_dir: Path) -> dict:
    from katarank.train.evaluate import evaluate_checkpoint
    return evaluate_checkpoint(str(ckpt_path), str(cache_dir), val_split="V",
                               loss_weights=_training_loss_weights())


# A val_loss win doesn't get vetoed for a merely noisy wiggle on these —
# only a regression past this tolerance blocks promotion.
_ACC_PM1_REGRESSION_TOLERANCE = 0.005      # 0.5 percentage points
_RATING_CORR_REGRESSION_TOLERANCE = 0.002


def _better(candidate: dict, live: dict) -> tuple[bool, str]:
    """Candidate is promoted iff its holdout val_loss is strictly lower
    than the live model's AND it doesn't regress the two practical-facing
    metrics beyond a small tolerance.

    val_loss is the actual training objective — a confidence-weighted
    ordinal likelihood (RankAnchorLoss) plus rating MSE plus Bradley-Terry
    pairwise ordering — so "more confident AND correct" already lowers it
    and "more confident but wrong" already raises it; it's a proper
    scoring rule, not a bare accuracy count. That's why it stays the
    single PRIMARY criterion rather than folding rank_acc_pm1/rating_corr
    into a weighted composite: those two are already >96%/99% (near
    ceiling) and highly correlated with each other, so inventing weights
    across them adds fragility (noise-sensitive tie-breaking) without a
    real signal val_loss doesn't already carry.

    What val_loss alone CAN'T catch: a reweighting artifact where val_loss
    improves but a practical accuracy measure someone actually reads
    regresses — hence the regression guardrail on rank_acc_pm1/rating_corr
    (not used to promote a worse-val_loss candidate, only to veto
    promoting a better-val_loss one that regressed elsewhere).

    Returns (promote, reason) — reason is recorded in model_versions.notes
    either way, including a val_loss win that got vetoed by the guardrail.
    """
    cand_loss = candidate.get("val_loss", float("inf"))
    live_loss = live.get("val_loss", float("inf"))
    if cand_loss >= live_loss:
        return False, f"val_loss {cand_loss:.4f} >= live val_loss {live_loss:.4f}"

    cand_acc, live_acc = candidate.get("rank_acc_pm1"), live.get("rank_acc_pm1")
    if cand_acc is not None and live_acc is not None and \
            cand_acc < live_acc - _ACC_PM1_REGRESSION_TOLERANCE:
        return False, (f"val_loss improved ({cand_loss:.4f} < {live_loss:.4f}) but "
                       f"rank_acc_pm1 regressed {live_acc:.4f} -> {cand_acc:.4f} "
                       f"(tolerance {_ACC_PM1_REGRESSION_TOLERANCE})")

    cand_corr, live_corr = candidate.get("rating_corr"), live.get("rating_corr")
    if cand_corr is not None and live_corr is not None and \
            cand_corr < live_corr - _RATING_CORR_REGRESSION_TOLERANCE:
        return False, (f"val_loss improved ({cand_loss:.4f} < {live_loss:.4f}) but "
                       f"rating_corr regressed {live_corr:.4f} -> {cand_corr:.4f} "
                       f"(tolerance {_RATING_CORR_REGRESSION_TOLERANCE})")

    return True, f"val_loss {cand_loss:.4f} < live val_loss {live_loss:.4f}, no guardrail regression"


# ── Promotion ─────────────────────────────────────────────────────────────────

def _archive_live_checkpoint(conn) -> str | None:
    """Copy the checkpoint about to be replaced into ARCHIVE_DIR before
    _promote overwrites LIVE_CKPT in place, keyed by its own
    model_versions.version — otherwise a promotion is destructive:
    the previous live weights would only survive as a metrics row with
    no way to actually roll back to them. Returns the archived version
    string (for logging), or None if there was nothing live yet to
    archive.
    """
    if not LIVE_CKPT.exists():
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM model_versions WHERE active = true AND mode = 'full'")
        row = cur.fetchone()
    version = row[0] if row else f"unversioned_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    dest_dir = ARCHIVE_DIR / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LIVE_CKPT, dest_dir / "best.pt")
    live_optim = LIVE_CKPT.parent / "optimizer_state.pt"
    if live_optim.exists():
        shutil.copy2(live_optim, dest_dir / "optimizer_state.pt")
    log.info("Archived outgoing live checkpoint (%s) to %s", version, dest_dir)
    return version


def _promote(conn, candidate_ckpt: Path, num_games: int, metrics: dict,
             katarank_url: str, reason: str) -> tuple[str, bool]:
    archived_version = _archive_live_checkpoint(conn)

    LIVE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_ckpt, LIVE_CKPT)
    # Carry the optimizer/scheduler state forward too — otherwise every
    # promotion resets continuity and the NEXT incremental run falls back
    # to a fresh optimizer against the just-promoted weights, the exact
    # cold-start disruption this mechanism exists to avoid.
    candidate_optim = candidate_ckpt.parent / "optimizer_state.pt"
    if candidate_optim.exists():
        shutil.copy2(candidate_optim, LIVE_CKPT.parent / "optimizer_state.pt")

    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    commit = at._git_commit()
    with conn.cursor() as cur:
        # Scoped to mode='full': this pipeline only ever trains the full
        # (1034-dim) model — deactivating every active row regardless of
        # mode would also flip off whatever lite model is currently live,
        # which this run never touched.
        cur.execute("UPDATE model_versions SET active = false WHERE active = true AND mode = 'full'")
        notes = f"promoted: {reason}"
        if archived_version:
            notes += f"; previous live checkpoint archived at {ARCHIVE_DIR / archived_version}"
        cur.execute("""
            INSERT INTO model_versions
                (version, artifact_path, architecture, feature_version, mode,
                 num_training_games, metrics, git_commit, notes, active, created_at)
            VALUES (%s, %s, %s, %s, 'full', %s, %s, %s, %s, true, now())
        """, (version, str(LIVE_CKPT), "transformer", "kab2-v3", num_games,
              json.dumps(metrics), commit, notes))
    conn.commit()
    served = at.reload_models(katarank_url, {"full": LIVE_CKPT})
    log.info("Promoted %s (now served: %s)", version, served)
    return version, served


def _discard(conn, num_games: int, candidate_metrics: dict, reason: str) -> str:
    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_discarded"
    commit = at._git_commit()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO model_versions
                (version, artifact_path, architecture, feature_version, mode,
                 num_training_games, metrics, git_commit, notes, active, created_at)
            VALUES (%s, %s, %s, %s, 'full', %s, %s, %s, %s, false, now())
        """, (version, str(CANDIDATE_DIR / "best.pt"), "transformer", "kab2-v3",
              num_games, json.dumps(candidate_metrics), commit, f"discarded: {reason}"))
    conn.commit()
    log.info("Discarded candidate (%s) — did not beat live model", version)
    return version


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="KataRank gated training session")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold", type=int, default=at.TRAIN_THRESHOLD)
    parser.add_argument("--db-dsn", default=at.DB_DSN)
    parser.add_argument("--cache-dir", type=Path, default=at.KAB2_CACHE)
    parser.add_argument("--katarank-url", default=KATARANK_URL_DEFAULT)
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train a fresh model instead of continuing from the "
                             "live checkpoint (full 150-epoch budget, no --resume, "
                             "no optimizer-state continuation). Both modes stay "
                             "available: default is incremental continuation.")
    args = parser.parse_args()

    t0 = time.time()
    _write_job(started_at=datetime.now(timezone.utc).isoformat(), result=None)

    state = at._load_state()
    katago_cfg = at._load_katago_config()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(args.db_dsn)
    try:
        _phase("checking")
        total = at.count_analyzed_games(conn)
        last = state.get("last_trained_count", 0)
        next_threshold = last + args.threshold
        if not args.force and total < next_threshold:
            _write_job(phase="idle", running=False,
                       result={"skipped": True, "reason": "below threshold",
                               "total": total, "next_threshold": next_threshold})
            log.info("Below threshold (%d < %d). Nothing to do.", total, next_threshold)
            return

        all_ids = at.get_analyzed_game_ids(conn)
        cached_ids = at.get_cached_game_ids(cache_dir)
        new_ids = all_ids - cached_ids

        if new_ids:
            _phase("exporting", new_games=len(new_ids))
            work_dir = Path(tempfile.mkdtemp(prefix="katarank_incr_"))
            sgf_dir, new_kab2_dir = work_dir / "sgfs", work_dir / "kab2"
            n_exported = at.export_sgfs_by_ids(conn, new_ids, sgf_dir)
            # End the read-only transaction before hours of GPU work: an 'idle in
            # transaction' connection holds locks on games for the whole session, so
            # any migration (ALTER TABLE) blocks — and then queues every later query.
            conn.commit()
            if n_exported:
                _phase("generating_kab2", new_games=n_exported)
                at.generate_kab2(sgf_dir, new_kab2_dir, katago_cfg)
                _phase("merging_cache")
                at.copy_new_npz(new_kab2_dir, cache_dir)
                at.merge_meta(cache_dir, new_kab2_dir)
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            log.info("All games already cached — skipping KAB2 generation")

        _phase("preparing_split")
        at.rebuild_meta(cache_dir)
        at.assign_train_val_split(cache_dir)

        with conn.cursor() as cur:
            cur.execute("SELECT metrics->>'best_val_loss' FROM model_versions "
                        "WHERE active = true AND mode = 'full'")
            row = cur.fetchone()
        conn.commit()  # same reason as after the export: training runs for hours
        live_val_loss = float(row[0]) if row and row[0] is not None else None

        _phase("training", staging_dir=str(CANDIDATE_DIR), from_scratch=args.from_scratch,
               epoch_progress=None, epoch_history=[], live_val_loss=live_val_loss)
        shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
        resume_from = (None if args.from_scratch else
                       str(LIVE_CKPT) if LIVE_CKPT.exists() else None)
        cmd = at.KATARANK_TRAIN + [
            "--data-dir", str(cache_dir),
            "--ckpt-dir", str(CANDIDATE_DIR),
        ]
        if resume_from:
            cmd += ["--resume", resume_from, "--patience", str(INCREMENTAL_PATIENCE)]
        import subprocess
        result = subprocess.run(cmd, cwd=str(at.PROJECT_ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"Training failed (exit code {result.returncode})")
        candidate_ckpt = CANDIDATE_DIR / "best.pt"
        if not candidate_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint produced at {candidate_ckpt}")

        _phase("evaluating")
        candidate_metrics = _evaluate(candidate_ckpt, cache_dir)
        live_metrics = (_evaluate(LIVE_CKPT, cache_dir) if LIVE_CKPT.exists()
                        else {"val_loss": float("inf")})
        log.info("candidate=%s  live=%s", candidate_metrics, live_metrics)

        n_total = len(at.get_cached_game_ids(cache_dir))
        promote, reason = _better(candidate_metrics, live_metrics)
        if promote:
            _phase("promoting")
            version, served = _promote(conn, candidate_ckpt, n_total, candidate_metrics,
                                       args.katarank_url, reason)
            outcome = "promoted"
        else:
            version = _discard(conn, n_total, candidate_metrics, reason)
            served = None
            outcome = "discarded"

        state["last_trained_count"] = total
        state.setdefault("runs", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "games": total, "cached_kab2": n_total, "new_generated": len(new_ids),
            "outcome": outcome, "version": version, "reason": reason,
            "candidate_metrics": candidate_metrics, "live_metrics": live_metrics,
        })
        at._save_state(state)

        _write_job(phase="idle", running=False, result={
            "skipped": False, "outcome": outcome, "version": version, "reason": reason,
            "served": served,
            "candidate_metrics": candidate_metrics, "live_metrics": live_metrics,
            "num_games": n_total, "elapsed_seconds": round(time.time() - t0, 1),
        })
        log.info("Done in %.1fs — outcome=%s", time.time() - t0, outcome)

    except Exception as e:  # noqa: BLE001
        log.exception("Training session failed")
        _write_job(phase="error", running=False, error=str(e))
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
