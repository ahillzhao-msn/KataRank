"""
KataRank — FastAPI Server
==========================
Exposes Go player rank assessment as a REST API.

Input: SGF strings, file paths, or directories.
Output: KAB2Output JSON (per-game rating + rank + confidence).

Start::

    uv run python -m katarank.api.server \\
        --model kata1-b18c384nbt.bin.gz \\
        --host 0.0.0.0 --port 8765

Endpoints
---------
POST /rank/string         rank from SGF content string
POST /rank/file           rank from SGF file path
POST /rank/batch          rank multiple SGFs (file paths or strings)
POST /rank/directory      rank all .sgf files in a directory
POST /review/string       rank + per-move records from SGF content string
POST /review/file         rank + per-move records from SGF file path
POST /review/batch        rank + per-move records for multiple SGFs
GET  /health              liveness check

Review endpoints return ReviewOutput = KAB2Output + 'moves' list
(docs/REVIEW_API_DESIGN.md); request schemas are identical to /rank/*.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def _require_fastapi():
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn required. Install:  uv add 'katarank[api]'"
        )


# ─── Request schemas ─────────────────────────────────────────────────────────

if _FASTAPI_AVAILABLE:
    class RankStringRequest(BaseModel):
        sgf: str
        mode: str = 'lite'
        min_moves: int = 10

    class RankFileRequest(BaseModel):
        path: str
        mode: str = 'lite'
        min_moves: int = 10

    class RankBatchRequest(BaseModel):
        items: List[str]          # file paths or SGF strings
        item_type: str = 'path'   # 'path' | 'string'
        mode: str = 'lite'
        min_moves: int = 10

    class RankDirectoryRequest(BaseModel):
        directory: str
        mode: str = 'lite'
        min_moves: int = 10

    class OwnershipStringRequest(BaseModel):
        sgf: str
        max_visits: int = 1

    class VariationStringRequest(BaseModel):
        sgf: str
        turn: int                                   # 0-based branch point
        extra_moves: List[List[str]] = []           # [['B','Q16'], ...]
        max_visits: int = 24
        human_profile: Optional[str] = None         # e.g. 'rank_5d'

    # ── Response schemas ──────────────────────────────────────────────────
    # Pydantic mirrors of schema.py's TypedDicts (KAB2Output / MoveRecord /
    # ReviewOutput). Their purpose is the OpenAPI contract: consumers like
    # gopredict read /openapi.json instead of importing this library.

    class KAB2OutputModel(BaseModel):
        """Whole-game verdict — one per game (schema.KAB2Output)."""
        game_id:      str
        metadata:     dict
        b_rating:     float
        w_rating:     float
        b_rank:       int                      # 0=20k … 28=9d, -1 unknown
        w_rank:       int
        b_confidence: float
        w_confidence: float
        b_rank_probs: Optional[List[float]] = None   # 29-dim distribution
        w_rank_probs: Optional[List[float]] = None

    class MoveRecordModel(BaseModel):
        """Per-move review record, mover's perspective (schema.MoveRecord)."""
        move_no:      int                      # 1-based global move number
        color:        str                      # 'B' | 'W'
        winrate:      float
        score_lead:   float
        score_stdev:  float
        policy_prior: float
        policy_rank:  int                      # 0 = engine's top choice
        win_delta:    float
        score_delta:  float
        ownership:    Optional[List[float]] = None  # 361 floats, stream mode only

    class ReviewOutputModel(KAB2OutputModel):
        """Verdict + per-move records (schema.ReviewOutput)."""
        moves: List[MoveRecordModel]


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app(
    katago_model: str,
    checkpoint_path: Optional[str] = None,
    katago_bin: Optional[str] = None,
    katago_config: Optional[str] = None,
    human_model: Optional[str] = None,
    device: Optional[str] = None,
    max_concurrency: int = 1,
    sgf_root: Optional[str] = None,
    persistent: bool = True,
    engine_mode: str = 'lite',
) -> 'FastAPI':
    """
    Create and configure the FastAPI application.

    Args:
        katago_model:    Path to KataGo model (.bin.gz)
        checkpoint_path: Optional KataRankModel checkpoint (.pt)
        katago_bin:      KataGo binary; auto-detected if None
        device:          'cpu', 'cuda', or None for auto
        max_concurrency: Max simultaneous katago analyses (default 1 = serialize)
        sgf_root:        If set, /rank/file and /rank/directory only accept
                         paths under this directory (path whitelist)
        persistent:      Keep one katago daemon loaded for the server's
                         lifetime (sub-second request latency). False spawns
                         a fresh process per request (model reload each time).
        engine_mode:     Daemon analysis mode, 'lite' or 'full'. Requests
                         asking for the other mode fall back to one-shot.
    """
    _require_fastapi()
    import threading
    from contextlib import asynccontextmanager

    from katarank.analysis_daemon import AnalysisDaemon
    from katarank.engine import KataGoEngine, PersistentKataGoEngine
    from katarank.workflow import InferenceWorkflow

    # ── Shared state ──────────────────────────────────────────────────────────

    # Batch analysis daemon (KAB2 protocol) — one GPU job at a time via engine_sem
    if persistent:
        engine = PersistentKataGoEngine(
            model       = katago_model,
            config      = katago_config,
            human_model = human_model,
            katago_bin  = katago_bin,
            mode        = engine_mode,
        )
        engine.start()   # pay the model load once, at server boot
    else:
        engine = KataGoEngine(
            model       = katago_model,
            config      = katago_config,
            human_model = human_model,
            katago_bin  = katago_bin,
        )

    # Online analysis daemon (JSON protocol) — separate process to avoid
    # blocking interactive queries behind long batch jobs on the GPU.
    analysis_daemon = AnalysisDaemon(
        katago_bin  = engine.katago_bin,
        model       = katago_model,
        config      = katago_config,
        human_model = human_model,
    )
    analysis_daemon.start()

    # Watchdog: restart dead daemons every 10 s without manual intervention.
    _watchdog_stop = threading.Event()

    def _watchdog():
        while not _watchdog_stop.wait(timeout=10):
            if not analysis_daemon.is_alive:
                tail = '\n'.join(list(analysis_daemon._stderr_tail)[-5:])
                logger.warning('AnalysisDaemon died — restarting%s',
                               f'\nlast stderr:\n{tail}' if tail else '')
                try:
                    analysis_daemon.restart()
                    logger.info('AnalysisDaemon restarted successfully')
                except Exception as exc:
                    logger.error('AnalysisDaemon restart failed: %s', exc)
            if persistent and isinstance(engine, PersistentKataGoEngine):
                if engine._proc is None or engine._proc.poll() is not None:
                    tail = ''.join(list(engine._stderr_tail)[-5:])
                    logger.warning('PersistentKataGoEngine died — restarting%s',
                                   f'\nlast stderr:\n{tail}' if tail else '')
                    try:
                        engine._proc = None
                        engine.start()
                        logger.info('PersistentKataGoEngine restarted successfully')
                    except Exception as exc:
                        logger.error('PersistentKataGoEngine restart failed: %s', exc)

    threading.Thread(target=_watchdog, daemon=True, name='katarank-watchdog').start()

    @asynccontextmanager
    async def _lifespan(app):
        yield
        _watchdog_stop.set()
        analysis_daemon.stop()
        if persistent:
            engine.close()

    app = FastAPI(
        title='KataRank API',
        description='Go player rank assessment via KataGo neural analysis',
        version='0.2.0',
        lifespan=_lifespan,
    )

    # Load rank model if checkpoint given
    inf_workflow: Optional[InferenceWorkflow] = None
    if checkpoint_path:
        from katarank.model import KataRankModel  # type: ignore
        rank_model = KataRankModel.load(checkpoint_path)
        inf_workflow = InferenceWorkflow(rank_model, engine, device=device)

    # Serialize katago runs: each analysis spawns a GPU-bound subprocess,
    # so unbounded concurrency would thrash the device.
    engine_sem = threading.Semaphore(max_concurrency)

    root = Path(sgf_root).resolve() if sgf_root else None

    def _check_path(p: str):
        """Reject paths outside sgf_root when a whitelist is configured."""
        if root is None:
            return
        try:
            ok = Path(p).resolve().is_relative_to(root)
        except (OSError, ValueError):
            ok = False
        if not ok:
            raise HTTPException(
                status_code=403,
                detail=f"Path outside allowed root {root}: {p}"
            )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get('/health')
    async def health():
        batch_alive = (not persistent) or (
            engine._proc is not None and engine._proc.poll() is None
        )
        analysis_alive = analysis_daemon.is_alive
        ok = batch_alive and analysis_alive
        return {
            'status':              'ok' if ok else 'degraded',
            'model_loaded':        inf_workflow is not None,
            'engine_persistent':   persistent,
            'engine_ready':        batch_alive,
            'analysis_daemon_ready': analysis_alive,
        }

    @app.post('/engine/reset')
    def engine_reset():
        """Soft reset: clear the daemon's NN caches (models stay loaded)."""
        if not persistent:
            raise HTTPException(status_code=400,
                                detail="No persistent engine (started with persistent=False)")
        try:
            with engine_sem:
                engine.soft_reset()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {'status': 'reset'}

    @app.post('/engine/restart')
    def engine_restart():
        """Hard restart: kill the daemon and reload models."""
        if not persistent:
            raise HTTPException(status_code=400,
                                detail="No persistent engine (started with persistent=False)")
        try:
            with engine_sem:
                engine.restart()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {'status': 'restarted'}

    # Endpoints below are plain `def`: FastAPI runs them in a worker thread,
    # keeping the event loop free while katago runs as a blocking subprocess.
    @app.post('/rank/string', response_model=KAB2OutputModel)
    def rank_string(req: RankStringRequest):
        """Rank players from an SGF content string. Returns KAB2Output JSON."""
        try:
            with engine_sem:
                results = _run_rank_strings(
                    engine, inf_workflow, [req.sgf], req.mode, req.min_moves
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not results:
            raise HTTPException(status_code=422,
                                detail='no output (too few moves?)')
        return dict(results[0])

    @app.post('/rank/file', response_model=KAB2OutputModel)
    def rank_file(req: RankFileRequest):
        """Rank players from an SGF file path. Returns KAB2Output JSON."""
        _check_path(req.path)
        if not Path(req.path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        try:
            with engine_sem:
                results = _run_rank_files(
                    engine, inf_workflow, [req.path], req.mode, req.min_moves
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not results:
            raise HTTPException(status_code=422,
                                detail='no output (too few moves?)')
        return dict(results[0])

    @app.post('/rank/batch', response_model=List[KAB2OutputModel])
    def rank_batch(req: RankBatchRequest):
        """Rank multiple SGFs. Items are file paths or SGF strings."""
        try:
            if req.item_type == 'string':
                with engine_sem:
                    results = _run_rank_strings(
                        engine, inf_workflow, req.items, req.mode, req.min_moves
                    )
            else:
                for p in req.items:
                    _check_path(p)
                missing = [p for p in req.items if not Path(p).exists()]
                if missing:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Files not found: {missing[:5]}"
                    )
                with engine_sem:
                    results = _run_rank_files(
                        engine, inf_workflow, req.items, req.mode, req.min_moves
                    )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return [dict(r) for r in results]

    @app.post('/review/string', response_model=ReviewOutputModel)
    def review_string(req: RankStringRequest):
        """Rank + per-move review records from an SGF content string.

        Ownership (361 floats) is attached via the persistent AnalysisDaemon,
        never persisted — stream/online mode only.
        """
        try:
            with engine_sem:
                results = _run_review_strings(
                    engine, inf_workflow, [req.sgf], req.mode, req.min_moves,
                    include_ownership=True,
                    analysis_daemon=analysis_daemon,
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not results:
            raise HTTPException(status_code=422,
                                detail='no output (too few moves?)')
        return dict(results[0])

    @app.post('/review/file', response_model=ReviewOutputModel)
    def review_file(req: RankFileRequest):
        """Rank + per-move review records from an SGF file path."""
        _check_path(req.path)
        if not Path(req.path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        try:
            with engine_sem:
                results = _run_review_files(
                    engine, inf_workflow, [req.path], req.mode, req.min_moves
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not results:
            raise HTTPException(status_code=422,
                                detail='no output (too few moves?)')
        return dict(results[0])

    @app.post('/review/batch', response_model=List[ReviewOutputModel])
    def review_batch(req: RankBatchRequest):
        """Rank + per-move review records for multiple SGFs."""
        try:
            if req.item_type == 'string':
                with engine_sem:
                    results = _run_review_strings(
                        engine, inf_workflow, req.items, req.mode, req.min_moves
                    )
            else:
                for p in req.items:
                    _check_path(p)
                missing = [p for p in req.items if not Path(p).exists()]
                if missing:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Files not found: {missing[:5]}"
                    )
                with engine_sem:
                    results = _run_review_files(
                        engine, inf_workflow, req.items, req.mode, req.min_moves
                    )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return [dict(r) for r in results]

    @app.post('/rank/directory', response_model=List[KAB2OutputModel])
    def rank_directory(req: RankDirectoryRequest):
        """Rank all .sgf files in a directory."""
        _check_path(req.directory)
        d = Path(req.directory)
        if not d.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory not found: {req.directory}")
        sgfs = sorted(str(p) for p in d.glob('*.sgf'))
        if not sgfs:
            raise HTTPException(status_code=404,
                                detail=f'No .sgf files found in {req.directory}')
        try:
            with engine_sem:
                results = _run_rank_files(
                    engine, inf_workflow, sgfs, req.mode, req.min_moves
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return [dict(r) for r in results]

    @app.post('/ownership/string')
    def ownership_string(req: OwnershipStringRequest):
        """Per-position ownership for each move in an SGF string.

        Routes to the persistent AnalysisDaemon (JSON protocol, no model
        reload) rather than the batch_analysis engine. Results are never
        persisted — stream/online mode only.

        Response: {"moves": [{"move_no": int, "ownership": [361 floats]}, ...]}
        move_no is 1-based (matches MoveRecord.move_no convention).
        """
        try:
            ownership_map = analysis_daemon.query_ownership(
                req.sgf, max_visits=req.max_visits
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        moves = [
            {'move_no': turn, 'ownership': own}
            for turn, own in sorted(ownership_map.items())
            if own is not None
        ]
        return {'moves': moves}

    @app.post('/variation/string')
    def variation_string(req: VariationStringRequest):
        """Top candidate moves for a what-if branch (trial-play research).

        Analyses the SGF truncated at `turn` plus `extra_moves`. With
        `human_profile` set (and the server started with --human-model),
        candidates reflect a player of that rank instead of full-strength
        KataGo. Routes to the persistent AnalysisDaemon; never persisted.
        """
        try:
            resp = analysis_daemon.query_variation(
                req.sgf, req.turn,
                extra_moves   = req.extra_moves,
                human_profile = req.human_profile,
                max_visits    = req.max_visits,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if resp is None:
            raise HTTPException(status_code=422,
                                detail='unparseable SGF or turn out of range')

        infos = resp.get('moveInfos', [])
        root  = resp.get('rootInfo', {})

        # humanSLProfile does NOT steer the search — KataGo only attaches
        # the human policy distribution. Rank-realistic play must therefore
        # be read from humanPolicy, not from moveInfos order.
        human_moves = []
        hp = resp.get('humanPolicy')
        if hp and len(hp) >= 361:
            cols = 'ABCDEFGHJKLMNOPQRST'
            top = sorted(range(361), key=lambda i: hp[i], reverse=True)[:5]
            human_moves = [
                {'move': f'{cols[i % 19]}{19 - i // 19}', 'prob': hp[i]}
                for i in top if hp[i] > 0
            ]

        return {
            'moves': [
                {'move': mi.get('move'), 'winrate': mi.get('winrate'),
                 'score_lead': mi.get('scoreLead'), 'visits': mi.get('visits'),
                 'order': mi.get('order'), 'prior': mi.get('prior')}
                for mi in infos
            ],
            'human_moves': human_moves,
            'root': {'winrate': root.get('winrate'),
                     'score_lead': root.get('scoreLead')},
        }

    return app


# ─── Helpers (shared with the CLI — see katarank/workflow.py) ────────────────

from katarank.workflow import (
    run_rank_files as _run_rank_files,
    run_rank_strings as _run_rank_strings,
    run_review_files as _run_review_files,
    run_review_strings as _run_review_strings,
)


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    _require_fastapi()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    parser = argparse.ArgumentParser(description='KataRank API Server')
    parser.add_argument('--model',       required=True,  help='KataGo model .bin.gz')
    parser.add_argument('--checkpoint',  default=None,   help='KataRankModel .pt')
    parser.add_argument('--katago-bin',  default=None,   help='KataGo binary path')
    parser.add_argument('--config',      default=None,   help='KataGo config .cfg')
    parser.add_argument('--human-model', default=None,   help='HumanSL model .bin.gz')
    parser.add_argument('--device',      default=None,   help='cpu / cuda')
    parser.add_argument('--host',        default='127.0.0.1')
    parser.add_argument('--port',        type=int, default=8765)
    parser.add_argument('--max-concurrency', type=int, default=1,
                        help='Max simultaneous katago analyses (default 1)')
    parser.add_argument('--sgf-root',    default=None,
                        help='Restrict /rank/file and /rank/directory to this directory')
    parser.add_argument('--no-persistent', action='store_true',
                        help='Spawn a fresh katago per request instead of a resident daemon')
    parser.add_argument('--engine-mode', default='lite', choices=['lite', 'full'],
                        help="Resident daemon's analysis mode (default lite)")
    args = parser.parse_args()

    app = create_app(
        katago_model    = args.model,
        checkpoint_path = args.checkpoint,
        katago_bin      = args.katago_bin,
        katago_config   = args.config,
        human_model     = args.human_model,
        device          = args.device,
        max_concurrency = args.max_concurrency,
        sgf_root        = args.sgf_root,
        persistent      = not args.no_persistent,
        engine_mode     = args.engine_mode,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
