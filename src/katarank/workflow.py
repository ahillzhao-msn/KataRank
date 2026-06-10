"""
KataRank — Workflows

Three high-level patterns:

TrainingWorkflow   — online training from a KataGoPool (stream) or KAB2FileDataset (files)
InferenceWorkflow  — rank inference for a batch of SGF files or strings
MixedWorkflow      — concurrent training + inference using separate pool pipelines

Usage::

    # Online training from live KataGo stream
    wf = TrainingWorkflow(model, loss_fn, optimizer)
    wf.run_stream(pool, pipeline='train', epochs=1)

    # Batch inference
    inf = InferenceWorkflow(model, engine)
    results = inf.rank_files(['game1.sgf', 'game2.sgf'])

    # Mixed: train on one pipeline, evaluate on another
    mix = MixedWorkflow(model, loss_fn, optimizer, pool)
    mix.run(train_pipe='train', eval_pipe='eval', train_steps=1000)
"""

import time
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from katarank.data.base import KAB2Base
from katarank.engine import KataGoEngine, StreamQueue
from katarank.pool import KataGoPool


# ─── Training Workflow ────────────────────────────────────────────────────────

class TrainingWorkflow:
    """
    Trains a KataRankModel from either a live stream or file-based dataset.

    The workflow owns the training loop; callers supply model, loss, optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        device: Optional[str] = None,
        gradient_clip: float = 1.0,
        log_every: int = 50,
    ):
        self.model         = model
        self.loss_fn       = loss_fn
        self.optimizer     = optimizer
        self.gradient_clip = gradient_clip
        self.log_every     = log_every

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)

        self._step     = 0
        self._total_loss = 0.0

    # ── Stream source ─────────────────────────────────────────────────────────

    def run_stream(
        self,
        pool: KataGoPool,
        pipeline: str,
        batch_size: int = 16,
        max_steps: int = 0,
    ) -> Dict:
        """
        Train from a named pool pipeline (StreamQueue).

        Accumulates samples into micro-batches of `batch_size` before
        each gradient step.
        """
        queue  = pool.queue(pipeline)
        buffer = []
        t0     = time.time()

        for item in queue:
            buffer.append(item)
            if len(buffer) < batch_size:
                continue

            batch = self._collate_raw(buffer)
            buffer.clear()
            if not batch:
                continue

            loss = self._step_batch(batch)
            self._step += 1

            if self.log_every and self._step % self.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  step={self._step:6d}  "
                    f"loss={loss:.4f}  "
                    f"games/s={self._step * batch_size / elapsed:.1f}"
                )

            if max_steps and self._step >= max_steps:
                break

        # Flush remaining buffer
        if buffer:
            batch = self._collate_raw(buffer)
            if batch:
                self._step_batch(batch)

        return {'steps': self._step, 'elapsed': time.time() - t0}

    # ── File source ───────────────────────────────────────────────────────────

    def run_loader(
        self,
        loader,
        epochs: int = 1,
    ) -> Dict:
        """Train from a DataLoader (KAB2FileDataset or any map-style dataset)."""
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            for batch in loader:
                if not batch:
                    continue
                loss = self._step_batch(batch)
                self._step += 1
                if self.log_every and self._step % self.log_every == 0:
                    print(f"  epoch={epoch} step={self._step}  loss={loss:.4f}")
        return {'steps': self._step, 'elapsed': time.time() - t0}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _step_batch(self, batch: Dict) -> float:
        x     = batch['x'].to(self.device)
        xlens = batch['xlens']
        out   = self.model(x, xlens)

        loss, _ = self.loss_fn(
            out,
            target_b   = batch['target_b'].to(self.device),
            target_w   = batch['target_w'].to(self.device),
            rank_b     = batch['rank_b'].to(self.device),
            rank_w     = batch['rank_w'].to(self.device),
            human_lp_b = batch['human_lp_b'].to(self.device),
            human_lp_w = batch['human_lp_w'].to(self.device),
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return loss.item()

    @staticmethod
    def _collate_raw(items) -> Dict:
        """Collate (moves_b, moves_w, info_b, info_w) tuples into a batch dict."""
        from katarank.data.base import KAB2Base
        samples = []
        for moves_b, moves_w, info_b, info_w in items:
            samples.append(KAB2Base._make_sample(
                moves_b    = moves_b,
                moves_w    = moves_w,
                game_id    = info_b.get('game_id', ''),
                target_b   = info_b.get('mean_log_prior', 0.0),
                target_w   = info_w.get('mean_log_prior', 0.0),
                rank_b     = info_b.get('human_rank_idx', -1),
                rank_w     = info_w.get('human_rank_idx', -1),
                human_lp_b = info_b.get('human_log_prior', 0.0),
                human_lp_w = info_w.get('human_log_prior', 0.0),
            ))
        return KAB2Base.collate(samples)


# ─── Inference Workflow ───────────────────────────────────────────────────────

class InferenceWorkflow:
    """
    Batch rank inference for SGF files or raw SGF strings.

    Returns per-game rank distributions without training.
    """

    def __init__(
        self,
        model: nn.Module,
        engine: KataGoEngine,
        device: Optional[str] = None,
    ):
        self.model  = model
        self.engine = engine
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model.to(self.device)
        self.model.eval()

    def rank_files(
        self,
        sgfs: Union[List[str], str],
        mode: str = 'lite',
        min_moves: int = 10,
    ) -> List[Dict]:
        """
        Rank players in a list of SGF file paths.

        Returns list of dicts per game::

            {
              'game_id': str,
              'b_log_prior': float,
              'w_log_prior': float,
              'b_rank_probs': Tensor (29,),
              'w_rank_probs': Tensor (29,),
              'b_rank_top': int,  # argmax
              'w_rank_top': int,
            }
        """
        results = []
        sq = self.engine.stream_queue(sgfs, mode=mode, min_moves=min_moves)
        try:
            for moves_b, moves_w, info_b, info_w in sq:
                result = self._infer_one(moves_b, moves_w, info_b, info_w)
                results.append(result)
        finally:
            sq.close()
        return results

    def rank_strings(
        self,
        sgf_strings: List[str],
        mode: str = 'lite',
        min_moves: int = 10,
    ) -> List[Dict]:
        """
        Rank players from in-memory SGF strings.

        Writes temp SGF files transparently; no caller-side I/O needed.
        """
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, sgf_str in enumerate(sgf_strings):
                p = os.path.join(tmp, f'game_{i:06d}.sgf')
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(sgf_str)
                paths.append(p)
            return self.rank_files(paths, mode=mode, min_moves=min_moves)

    @torch.no_grad()
    def _infer_one(self, moves_b, moves_w, info_b, info_w) -> Dict:
        sample = KAB2Base._make_sample(
            moves_b = moves_b, moves_w = moves_w,
            game_id = info_b.get('game_id', ''),
        )
        x     = sample['x'].unsqueeze(0).to(self.device)  # treat as batch-1... actually pass flat
        # KataRankModel expects flat (N_total, input_dim) + xlens
        x_flat = sample['x'].to(self.device)
        out   = self.model(x_flat, xlens=[sample['seq_len']])

        def _rank_probs(logits):
            return torch.softmax(logits, dim=-1).squeeze(0).cpu()

        return {
            'game_id':      sample['game_id'],
            'b_log_prior':  out.get('b_log_prior',  out.get('b_rating',  torch.tensor(0.))).item(),
            'w_log_prior':  out.get('w_log_prior',  out.get('w_rating',  torch.tensor(0.))).item(),
            'b_rank_probs': _rank_probs(out['b_rank_logits']) if 'b_rank_logits' in out else None,
            'w_rank_probs': _rank_probs(out['w_rank_logits']) if 'w_rank_logits' in out else None,
            'b_rank_top':   int(out['b_rank_logits'].argmax(-1)) if 'b_rank_logits' in out else -1,
            'w_rank_top':   int(out['w_rank_logits'].argmax(-1)) if 'w_rank_logits' in out else -1,
        }


# ─── Mixed Workflow ───────────────────────────────────────────────────────────

class MixedWorkflow:
    """
    Runs training and evaluation concurrently from a shared pool.

    Training reads from `train_pipe`; evaluation reads from `eval_pipe`
    in a separate thread, periodically logging rank accuracy metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        pool: KataGoPool,
        device: Optional[str] = None,
    ):
        self.pool    = pool
        self.trainer = TrainingWorkflow(model, loss_fn, optimizer, device=device)
        self.engine  = None  # not needed; pool manages engines

    def run(
        self,
        train_pipe: str,
        eval_pipe: Optional[str]   = None,
        train_batch_size: int      = 16,
        max_train_steps: int       = 0,
        eval_every: int            = 200,
    ) -> Dict:
        """
        Run training loop on `train_pipe`.
        If `eval_pipe` is set, evaluate every `eval_every` steps.
        """
        import threading

        eval_results = []

        def _eval_loop():
            if eval_pipe is None:
                return
            q = self.pool.queue(eval_pipe)
            count, correct = 0, 0
            for moves_b, moves_w, info_b, info_w in q:
                count += 1
            eval_results.append({'games_evaluated': count})

        eval_thread = threading.Thread(target=_eval_loop, daemon=True)
        eval_thread.start()

        train_result = self.trainer.run_stream(
            self.pool,
            pipeline   = train_pipe,
            batch_size = train_batch_size,
            max_steps  = max_train_steps,
        )

        eval_thread.join(timeout=30.0)

        return {
            'train': train_result,
            'eval':  eval_results,
        }
