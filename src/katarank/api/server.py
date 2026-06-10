"""
KataRank — FastAPI Server

Exposes rank inference and pool management as a REST API.
Other programs interact via HTTP: submit SGF strings, get rank results.

Start::

    uv run python -m katarank.api.server \
        --model kata1-b18c384nbt.bin.gz \
        --checkpoint nets/katarank/best.pt \
        --host 0.0.0.0 --port 8765

Or programmatically::

    app = create_app(model_path, checkpoint_path, katago_model)
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8765)

Endpoints
---------
POST /rank/string        rank players from an SGF string
POST /rank/file          rank players from an SGF file path
POST /rank/batch         rank a list of SGF file paths
GET  /pool/status        status of active KataGo pipelines
POST /pool/start         start a new pipeline
DELETE /pool/{name}      stop a pipeline
GET  /health             liveness check
"""

from __future__ import annotations

import argparse
import tempfile
import os
from pathlib import Path
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def _require_fastapi():
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn are required for the API server.\n"
            "Install with:  uv add 'katarank[api]'"
        )


# ─── Request/Response schemas ─────────────────────────────────────────────────

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
        paths: List[str]
        mode: str = 'lite'
        min_moves: int = 10

    class StartPipelineRequest(BaseModel):
        name: str
        sgfs: str        # CSV list file path or SGF directory glob
        mode: str = 'full'
        buffer_size: int = 32
        min_moves: int = 10


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app(
    katago_model: str,
    checkpoint_path: Optional[str] = None,
    katago_bin: Optional[str] = None,
    katago_config: Optional[str] = None,
    human_model: Optional[str] = None,
    device: Optional[str] = None,
) -> 'FastAPI':
    """
    Create and configure the FastAPI application.

    Args:
        katago_model:    Path to KataGo model (.bin.gz)
        checkpoint_path: KataRankModel checkpoint (.pt); optional
        katago_bin:      KataGo binary; auto-detected if None
        device:          'cpu', 'cuda', or None for auto
    """
    _require_fastapi()

    from katarank.engine import KataGoEngine
    from katarank.pool import KataGoPool
    from katarank.workflow import InferenceWorkflow

    app = FastAPI(
        title='KataRank API',
        description='Go player rank assessment via KataGo neural analysis',
        version='0.1.0',
    )

    # ── Shared state ──────────────────────────────────────────────────────────

    engine = KataGoEngine(
        model      = katago_model,
        config     = katago_config,
        human_model= human_model,
        katago_bin = katago_bin,
    )
    pool = KataGoPool(
        model      = katago_model,
        config     = katago_config,
        human_model= human_model,
        katago_bin = katago_bin,
    )

    # Load rank model if checkpoint given
    rank_model = None
    if checkpoint_path:
        from katarank.model import KataRankModel
        rank_model = KataRankModel.load(checkpoint_path)

    inf_workflow: Optional[InferenceWorkflow] = None
    if rank_model is not None:
        inf_workflow = InferenceWorkflow(rank_model, engine, device=device)

    def _require_model():
        if inf_workflow is None:
            raise HTTPException(
                status_code=503,
                detail="No KataRankModel checkpoint loaded. "
                       "Start server with --checkpoint path/to/best.pt"
            )
        return inf_workflow

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'model_loaded': rank_model is not None}

    @app.post('/rank/string')
    async def rank_string(req: RankStringRequest):
        wf = _require_model()
        try:
            results = wf.rank_strings([req.sgf], mode=req.mode, min_moves=req.min_moves)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return results[0] if results else {'error': 'no output (too few moves?)'}

    @app.post('/rank/file')
    async def rank_file(req: RankFileRequest):
        wf = _require_model()
        if not Path(req.path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        try:
            results = wf.rank_files([req.path], mode=req.mode, min_moves=req.min_moves)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return results[0] if results else {'error': 'no output'}

    @app.post('/rank/batch')
    async def rank_batch(req: RankBatchRequest):
        wf = _require_model()
        missing = [p for p in req.paths if not Path(p).exists()]
        if missing:
            raise HTTPException(status_code=404, detail=f"Files not found: {missing[:5]}")
        try:
            return wf.rank_files(req.paths, mode=req.mode, min_moves=req.min_moves)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/pool/status')
    async def pool_status():
        return pool.status()

    @app.post('/pool/start')
    async def pool_start(req: StartPipelineRequest):
        try:
            pool.start(
                req.name,
                sgfs        = req.sgfs,
                mode        = req.mode,
                buffer_size = req.buffer_size,
                min_moves   = req.min_moves,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {'started': req.name}

    @app.delete('/pool/{name}')
    async def pool_stop(name: str):
        try:
            pool.stop(name, wait=False)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")
        return {'stopped': name}

    # ── Shutdown hook ─────────────────────────────────────────────────────────
    @app.on_event('shutdown')
    async def _shutdown():
        pool.close_all(wait=False)

    return app


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    _require_fastapi()
    parser = argparse.ArgumentParser(description='KataRank API Server')
    parser.add_argument('--model',      required=True,  help='KataGo model .bin.gz')
    parser.add_argument('--checkpoint', default=None,   help='KataRankModel .pt')
    parser.add_argument('--katago-bin', default=None,   help='KataGo binary path')
    parser.add_argument('--config',     default=None,   help='KataGo config .cfg')
    parser.add_argument('--human-model',default=None,   help='HumanSL model .bin.gz')
    parser.add_argument('--device',     default=None,   help='cpu / cuda')
    parser.add_argument('--host',       default='127.0.0.1')
    parser.add_argument('--port',       type=int, default=8765)
    parser.add_argument('--reload',     action='store_true')
    args = parser.parse_args()

    app = create_app(
        katago_model    = args.model,
        checkpoint_path = args.checkpoint,
        katago_bin      = args.katago_bin,
        katago_config   = args.config,
        human_model     = args.human_model,
        device          = args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
