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
GET  /health              liveness check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

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

    from katarank.engine import KataGoEngine, PersistentKataGoEngine
    from katarank.workflow import InferenceWorkflow

    app = FastAPI(
        title='KataRank API',
        description='Go player rank assessment via KataGo neural analysis',
        version='0.2.0',
    )

    # ── Shared state ──────────────────────────────────────────────────────────

    if persistent:
        engine = PersistentKataGoEngine(
            model       = katago_model,
            config      = katago_config,
            human_model = human_model,
            katago_bin  = katago_bin,
            mode        = engine_mode,
        )
        engine.start()   # pay the model load once, at server boot
        app.add_event_handler('shutdown', engine.close)
    else:
        engine = KataGoEngine(
            model       = katago_model,
            config      = katago_config,
            human_model = human_model,
            katago_bin  = katago_bin,
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
        alive = True
        if persistent:
            alive = engine._proc is not None and engine._proc.poll() is None
        return {
            'status': 'ok' if alive else 'engine_down',
            'model_loaded': inf_workflow is not None,
            'engine_persistent': persistent,
            'engine_ready': alive,
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
    @app.post('/rank/string')
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
            return {'error': 'no output (too few moves?)'}
        return dict(results[0])

    @app.post('/rank/file')
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
            return {'error': 'no output'}
        return dict(results[0])

    @app.post('/rank/batch')
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

    @app.post('/rank/directory')
    def rank_directory(req: RankDirectoryRequest):
        """Rank all .sgf files in a directory."""
        _check_path(req.directory)
        d = Path(req.directory)
        if not d.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory not found: {req.directory}")
        sgfs = sorted(str(p) for p in d.glob('*.sgf'))
        if not sgfs:
            return {'error': f'No .sgf files found in {req.directory}'}
        try:
            with engine_sem:
                results = _run_rank_files(
                    engine, inf_workflow, sgfs, req.mode, req.min_moves
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return [dict(r) for r in results]

    return app


# ─── Helpers (shared with the CLI — see katarank/workflow.py) ────────────────

from katarank.workflow import (
    run_rank_files as _run_rank_files,
    run_rank_strings as _run_rank_strings,
)


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    _require_fastapi()
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
