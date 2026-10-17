"""
KataRank — Gated distillation session. Lite counterpart of train_session.py:
same staging + evaluation-gate + promote/discard/archive mechanics, applied
to the lite (10-dim scalar-only) student model instead of the full model.

    1-4. (identical to train_session.py / auto_train.py: export new SGFs,
         generate KAB2, merge cache, rebuild _meta.csv + T/V split)
    5. Distill into nets/_candidate_lite/, teacher = the CURRENT live full
       checkpoint (nets/katarank/best.pt) — always distills from whatever
       full model is live right now, not a pinned version. Resumes student
       weights + optimizer/scheduler state from the live lite checkpoint
       unless --from-scratch.
    6. Evaluate BOTH the live lite checkpoint and the candidate on the same
       fresh holdout split, sliced to the lite model's 10-dim input (see
       evaluate.py's input_dim param) — same KataRankLoss weighting the
       full-model gate uses, NOT the distillation loss, so "did this
       actually get better" is judged on the same yardstick as any other
       KataRankModel checkpoint, independent of distillation temperature.
    7. Candidate strictly better (lower val_loss, no regression on
       rank_acc_pm1/rating_corr beyond tolerance) → promote: copy into the
       live lite checkpoint path, record model_versions (mode='lite',
       active=true, deactivating the previous lite row), and have the
       running server reload it (auto_train.reload_models, sha256-verified).
       Candidate not better → discard: audit-only model_versions row,
       live lite checkpoint untouched.

Progress is written to ~/.katarank/distill_job.json (separate from
train_job.json — full and lite runs are triggered independently and
each needs its own phase/progress state). Usage:

    uv run python scripts/distill_session.py
    uv run python scripts/distill_session.py --force --threshold 500
    uv run python scripts/distill_session.py --promote-candidate "why"   # see _promote_override
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

log = logging.getLogger("distill_session")

JOB_FILE = Path.home() / ".katarank" / "distill_job.json"
LIVE_CKPT = at.PROJECT_ROOT / "nets" / "katarank" / "best.pt"            # teacher
LIVE_LITE_CKPT = at.PROJECT_ROOT / "nets" / "katarank_lite" / "best_lite.pt"
CANDIDATE_LITE_DIR = at.PROJECT_ROOT / "nets" / "_candidate_lite"
ARCHIVE_LITE_DIR = at.PROJECT_ROOT / "nets" / "katarank_lite_archive"
KATARANK_URL_DEFAULT = "http://127.0.0.1:8765"
STUDENT_INPUT_DIM = 10

# Same rationale as train_session.py's INCREMENTAL_PATIENCE — a --resume run
# starts from an already-converged student, so genuine improvement should
# show within a handful of epochs.
INCREMENTAL_PATIENCE = 5


# ── Job status file (mirrors train_session.py's _write_job/_phase) ──────────

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


# ── Evaluation gate (same criteria as train_session.py, lite-dim aware) ─────

def _training_loss_weights() -> dict:
    import yaml
    config_path = at.PROJECT_ROOT / "src" / "katarank" / "train" / "config_kata_native.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("training", {}).get("loss_weights", {})


def _evaluate(ckpt_path: Path, cache_dir: Path) -> dict:
    from katarank.train.evaluate import evaluate_checkpoint
    return evaluate_checkpoint(str(ckpt_path), str(cache_dir), val_split="V",
                               loss_weights=_training_loss_weights(),
                               input_dim=STUDENT_INPUT_DIM)


_ACC_PM1_REGRESSION_TOLERANCE = 0.005
_RATING_CORR_REGRESSION_TOLERANCE = 0.002


def _better(candidate: dict, live: dict) -> tuple[bool, str]:
    """Identical criteria to train_session.py's _better() — see its
    docstring for the full rationale. Duplicated rather than imported
    because the two files are meant to stay independently readable."""
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

def _archive_live_lite_checkpoint(conn) -> str | None:
    if not LIVE_LITE_CKPT.exists():
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM model_versions WHERE active = true AND mode = 'lite'")
        row = cur.fetchone()
    version = row[0] if row else f"unversioned_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    dest_dir = ARCHIVE_LITE_DIR / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LIVE_LITE_CKPT, dest_dir / "best_lite.pt")
    live_optim = LIVE_LITE_CKPT.parent / "optimizer_state.pt"
    if live_optim.exists():
        shutil.copy2(live_optim, dest_dir / "optimizer_state.pt")
    log.info("Archived outgoing live lite checkpoint (%s) to %s", version, dest_dir)
    return version


def _promote(conn, candidate_ckpt: Path, num_games: int, metrics: dict,
             katarank_url: str, reason: str, teacher_version: str | None) -> tuple[str, bool]:
    archived_version = _archive_live_lite_checkpoint(conn)

    LIVE_LITE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_ckpt, LIVE_LITE_CKPT)
    candidate_optim = candidate_ckpt.parent / "optimizer_state.pt"
    if candidate_optim.exists():
        shutil.copy2(candidate_optim, LIVE_LITE_CKPT.parent / "optimizer_state.pt")

    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    commit = at._git_commit()
    with conn.cursor() as cur:
        cur.execute("UPDATE model_versions SET active = false WHERE active = true AND mode = 'lite'")
        notes = f"promoted: {reason}; teacher={teacher_version or LIVE_CKPT}"
        if archived_version:
            notes += f"; previous live lite checkpoint archived at {ARCHIVE_LITE_DIR / archived_version}"
        cur.execute("""
            INSERT INTO model_versions
                (version, artifact_path, architecture, feature_version, mode,
                 num_training_games, metrics, git_commit, notes, active, created_at)
            VALUES (%s, %s, %s, %s, 'lite', %s, %s, %s, %s, true, now())
        """, (version, str(LIVE_LITE_CKPT), "transformer", "kab2-v3", num_games,
              json.dumps(metrics), commit, notes))
    conn.commit()
    served = at.reload_models(katarank_url, {"lite": LIVE_LITE_CKPT})
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
            VALUES (%s, %s, %s, %s, 'lite', %s, %s, %s, %s, false, now())
        """, (version, str(CANDIDATE_LITE_DIR / "best_lite.pt"), "transformer", "kab2-v3",
              num_games, json.dumps(candidate_metrics), commit, f"discarded: {reason}"))
    conn.commit()
    log.info("Discarded candidate (%s) — did not beat live lite model", version)
    return version


def _promote_override(args, reason: str) -> None:
    """Promote the candidate the last session left in CANDIDATE_LITE_DIR even
    though the gate rejected it — lite counterpart of train_session.py's
    _promote_override, for a deliberate, recorded decision. Both lite
    checkpoints are evaluated afresh on the current V split, and the real
    metrics, the gate's own verdict and the given reason all go into
    model_versions.notes and the lite_runs log. The gate itself is
    unchanged; nothing calls this automatically.
    """
    candidate_ckpt = CANDIDATE_LITE_DIR / "best_lite.pt"
    if not candidate_ckpt.exists():
        raise SystemExit(f"No candidate at {candidate_ckpt}")
    job = json.loads(JOB_FILE.read_text(encoding="utf-8")) if JOB_FILE.exists() else {}
    if job.get("running"):
        # The running session is still writing CANDIDATE_LITE_DIR.
        raise SystemExit(f"A distillation session is running ({JOB_FILE}); wait for it to finish")
    # The candidate was distilled by the session recorded in JOB_FILE.
    teacher_version = job.get("teacher_version")
    candidate_metrics = _evaluate(candidate_ckpt, args.cache_dir)
    live_metrics = (_evaluate(LIVE_LITE_CKPT, args.cache_dir) if LIVE_LITE_CKPT.exists()
                    else {"val_loss": float("inf")})
    log.info("candidate=%s  live=%s", candidate_metrics, live_metrics)
    gate_promote, gate_reason = _better(candidate_metrics, live_metrics)
    note = f"manual override ({reason}); gate {'passed' if gate_promote else 'rejected'}: {gate_reason}"

    n_total = len(at.get_cached_game_ids(args.cache_dir))
    conn = psycopg2.connect(args.db_dsn)
    try:
        version, served = _promote(conn, candidate_ckpt, n_total, candidate_metrics,
                                   args.katarank_url, note, teacher_version)
    finally:
        conn.close()
    state = at._load_state()
    state.setdefault("lite_runs", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cached_kab2": n_total, "teacher_version": teacher_version,
        "outcome": "promoted_override", "version": version, "reason": note, "served": served,
        "candidate_metrics": candidate_metrics, "live_metrics": live_metrics,
    })
    at._save_state(state)
    log.info("Override-promoted %s (served=%s): %s", version, served, note)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="KataRank gated distillation session")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold", type=int, default=at.TRAIN_THRESHOLD)
    parser.add_argument("--db-dsn", default=at.DB_DSN)
    parser.add_argument("--cache-dir", type=Path, default=at.KAB2_CACHE)
    parser.add_argument("--katarank-url", default=KATARANK_URL_DEFAULT)
    parser.add_argument("--teacher", default=None,
                        help="Override teacher checkpoint (default: current live full model)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train a fresh student instead of continuing from the "
                             "live lite checkpoint. Both modes stay available: "
                             "default is incremental continuation.")
    parser.add_argument("--promote-candidate", metavar="REASON",
                        help="Don't distill: promote the candidate the last session "
                             "left in nets/_candidate_lite over the gate's verdict, "
                             "recording REASON and both models' fresh metrics.")
    args = parser.parse_args()
    if args.promote_candidate:
        _promote_override(args, args.promote_candidate)
        return

    t0 = time.time()
    _write_job(started_at=datetime.now(timezone.utc).isoformat(), result=None)

    # Separate counter from train_session.py's "last_trained_count" in the
    # SAME state file — full and lite are triggered independently off the
    # same game-count signal but must not clobber each other's watermark.
    state = at._load_state()
    katago_cfg = at._load_katago_config()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = Path(args.teacher) if args.teacher else LIVE_CKPT
    if not teacher_path.exists():
        _write_job(phase="error", running=False,
                   error=f"No teacher checkpoint at {teacher_path}")
        log.error("No teacher checkpoint at %s", teacher_path)
        return

    conn = psycopg2.connect(args.db_dsn)
    try:
        _phase("checking")
        total = at.count_analyzed_games(conn)
        last = state.get("last_lite_trained_count", 0)
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
            cur.execute("SELECT version FROM model_versions WHERE active = true AND mode = 'full'")
            row = cur.fetchone()
        teacher_version = row[0] if row else None

        with conn.cursor() as cur:
            cur.execute("SELECT metrics->>'best_val_loss' FROM model_versions "
                        "WHERE active = true AND mode = 'lite'")
            row = cur.fetchone()
        live_val_loss = float(row[0]) if row and row[0] is not None else None

        _phase("distilling", staging_dir=str(CANDIDATE_LITE_DIR), from_scratch=args.from_scratch,
               teacher=str(teacher_path), teacher_version=teacher_version,
               epoch_progress=None, epoch_history=[], live_val_loss=live_val_loss)
        shutil.rmtree(CANDIDATE_LITE_DIR, ignore_errors=True)
        resume_from = (None if args.from_scratch else
                       str(LIVE_LITE_CKPT) if LIVE_LITE_CKPT.exists() else None)
        cmd = at.KATARANK_DISTILL + [
            "--teacher", str(teacher_path),
            "--data-dir", str(cache_dir),
            "--ckpt-dir", str(CANDIDATE_LITE_DIR),
        ]
        if resume_from:
            cmd += ["--resume", resume_from, "--patience", str(INCREMENTAL_PATIENCE)]
        import subprocess
        result = subprocess.run(cmd, cwd=str(at.PROJECT_ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"Distillation failed (exit code {result.returncode})")
        candidate_ckpt = CANDIDATE_LITE_DIR / "best_lite.pt"
        if not candidate_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint produced at {candidate_ckpt}")

        _phase("evaluating")
        candidate_metrics = _evaluate(candidate_ckpt, cache_dir)
        live_metrics = (_evaluate(LIVE_LITE_CKPT, cache_dir) if LIVE_LITE_CKPT.exists()
                        else {"val_loss": float("inf")})
        log.info("candidate=%s  live=%s", candidate_metrics, live_metrics)

        n_total = len(at.get_cached_game_ids(cache_dir))
        promote, reason = _better(candidate_metrics, live_metrics)
        if promote:
            _phase("promoting")
            version, served = _promote(conn, candidate_ckpt, n_total, candidate_metrics,
                                       args.katarank_url, reason, teacher_version)
            outcome = "promoted"
        else:
            version = _discard(conn, n_total, candidate_metrics, reason)
            served = None
            outcome = "discarded"

        state["last_lite_trained_count"] = total
        state.setdefault("lite_runs", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "games": total, "cached_kab2": n_total, "new_generated": len(new_ids),
            "teacher_version": teacher_version,
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
        log.exception("Distillation session failed")
        _write_job(phase="error", running=False, error=str(e))
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
