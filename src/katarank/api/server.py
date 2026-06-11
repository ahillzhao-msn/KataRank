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
) -> 'FastAPI':
    """
    Create and configure the FastAPI application.

    Args:
        katago_model:    Path to KataGo model (.bin.gz)
        checkpoint_path: Optional KataRankModel checkpoint (.pt)
        katago_bin:      KataGo binary; auto-detected if None
        device:          'cpu', 'cuda', or None for auto
    """
    _require_fastapi()

    from katarank.engine import KataGoEngine
    from katarank.workflow import InferenceWorkflow
    from katarank.schema import (
        KAB2Output, output_to_json, rank_idx_to_str,
    )

    app = FastAPI(
        title='KataRank API',
        description='Go player rank assessment via KataGo neural analysis',
        version='0.2.0',
    )

    # ── Shared state ──────────────────────────────────────────────────────────

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
        return {
            'status': 'ok',
            'model_loaded': inf_workflow is not None,
            'engine_ready': True,
        }

    @app.post('/rank/string')
    async def rank_string(req: RankStringRequest):
        """Rank players from an SGF content string. Returns KAB2Output JSON."""
        try:
            results = _run_rank_strings(
                engine, inf_workflow, [req.sgf], req.mode, req.min_moves
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not results:
            return {'error': 'no output (too few moves?)'}
        return dict(results[0])

    @app.post('/rank/file')
    async def rank_file(req: RankFileRequest):
        """Rank players from an SGF file path. Returns KAB2Output JSON."""
        if not Path(req.path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        try:
            results = _run_rank_files(
                engine, inf_workflow, [req.path], req.mode, req.min_moves
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not results:
            return {'error': 'no output'}
        return dict(results[0])

    @app.post('/rank/batch')
    async def rank_batch(req: RankBatchRequest):
        """Rank multiple SGFs. Items are file paths or SGF strings."""
        try:
            if req.item_type == 'string':
                results = _run_rank_strings(
                    engine, inf_workflow, req.items, req.mode, req.min_moves
                )
            else:
                missing = [p for p in req.items if not Path(p).exists()]
                if missing:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Files not found: {missing[:5]}"
                    )
                results = _run_rank_files(
                    engine, inf_workflow, req.items, req.mode, req.min_moves
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return [dict(r) for r in results]

    @app.post('/rank/directory')
    async def rank_directory(req: RankDirectoryRequest):
        """Rank all .sgf files in a directory."""
        d = Path(req.directory)
        if not d.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory not found: {req.directory}")
        sgfs = sorted(str(p) for p in d.glob('*.sgf'))
        if not sgfs:
            return {'error': f'No .sgf files found in {req.directory}'}
        try:
            results = _run_rank_files(
                engine, inf_workflow, sgfs, req.mode, req.min_moves
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return [dict(r) for r in results]

    return app


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_rank_files(
    engine, inf_workflow, paths: List[str], mode: str, min_moves: int
) -> List[KAB2Output]:
    """Rank from SGF file paths using lite mode (stream)."""
    results = []
    for side, moves, info in engine.stream_games(
        sgf_paths=paths, mode=mode, min_moves=min_moves
    ):
        _accumulate_result(results, side, moves, info)
    return results


def _run_rank_strings(
    engine, inf_workflow, strings: List[str], mode: str, min_moves: int
) -> List[KAB2Output]:
    """Rank from SGF content strings using lite mode (stream)."""
    results = []
    for side, moves, info in engine.stream_games(
        sgf_strings=strings, mode=mode, min_moves=min_moves
    ):
        _accumulate_result(results, side, moves, info)
    return results


def _accumulate_result(results, side, moves, info):
    """Accumulate B/W pairs into KAB2Output."""
    from katarank.schema import KAB2Output, rank_idx_to_str

    # Find or create entry for this game (identified by game_id or sequential)
    if not results or results[-1].get('w_rating') is not None:
        # Need a new game — but we don't have game_id from lite stream info
        # Use sequential index
        idx = len(results)
        results.append(KAB2Output(
            game_id=info.get('source', f'game_{idx:04d}'),
            metadata={},
            b_rating=0.0, w_rating=0.0,
            b_rank=-1, w_rank=-1,
            b_confidence=0.0, w_confidence=0.0,
            b_rank_probs=None, w_rank_probs=None,
        ))

    entry = results[-1]
    if side == 'B':
        entry['b_rating'] = float(info['mean_log_prior'])
        entry['b_rank'] = info['human_rank_idx']
        entry['b_confidence'] = 1.0 - abs(info['mean_log_prior']) / 10.0
    elif side == 'W':
        entry['w_rating'] = float(info['mean_log_prior'])
        entry['w_rank'] = info['human_rank_idx']
        entry['w_confidence'] = 1.0 - abs(info['mean_log_prior']) / 10.0


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
    args = parser.parse_args()

    app = create_app(
        katago_model    = args.model,
        checkpoint_path = args.checkpoint,
        katago_bin      = args.katago_bin,
        katago_config   = args.config,
        human_model     = args.human_model,
        device          = args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
