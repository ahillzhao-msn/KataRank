"""
KataRank — Workflows

Two high-level patterns:

TrainingWorkflow   — online training from any stream of (moves_b, moves_w, info_b, info_w)
InferenceWorkflow  — rank inference for a batch of SGF files or strings

Usage::

    # Online training from a live stream
    engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')
    stream = engine.stream_to_tensors(sgfs, mode='full')
    wf = TrainingWorkflow(model, loss_fn, optimizer)
    wf.run_stream(stream, batch_size=16, max_steps=1000)

    # Batch inference
    inf = InferenceWorkflow(model, engine)
    results = inf.rank_files(['game1.sgf', 'game2.sgf'])
"""

import time
from typing import Callable, Dict, Iterator, List, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from katarank.data.katago_native import KAB2Sample, kab2_make_sample, kab2_collate


# ─── Inference result type ────────────────────────────────────────────────────

class RankResult(TypedDict):
    """Per-game inference output from InferenceWorkflow."""
    game_id:      str
    b_log_prior:  float            # raw rating signal for Black
    w_log_prior:  float            # raw rating signal for White
    b_rank_probs: Optional[torch.Tensor]  # (29,) softmax over rank classes
    w_rank_probs: Optional[torch.Tensor]
    b_rank_top:   int              # argmax rank index (0=20k … 28=9d), -1 if unavailable
    w_rank_top:   int


# ─── Training Workflow ────────────────────────────────────────────────────────

class TrainingWorkflow:
    """
    Trains a KataRankModel from any iterator of (moves_b, moves_w, info_b, info_w).

    The workflow owns the training loop; callers supply model, loss, optimizer,
    and an iterator (e.g. from KataGoEngine.stream_to_tensors() or a DataLoader).
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

    def run_stream(
        self,
        stream: Iterator,
        batch_size: int = 16,
        max_steps: int = 0,
    ) -> Dict:
        """
        Train from any iterator yielding (moves_b, moves_w, info_b, info_w).

        Accumulates samples into micro-batches of `batch_size` before
        each gradient step.

        Args:
            stream:    Iterator over (moves_b, moves_w, info_b, info_w) tuples.
                       Can be:
                         - KataGoEngine.stream_to_tensors()  (torch tensors)
                         - KAB2StreamDataset                  (KAB2Sample)
                         - Any custom iterable with the same tuple shape
            batch_size: Number of games per gradient step
            max_steps:  Max gradient steps (0 = unlimited)
        """
        buffer = []
        t0     = time.time()

        for item in stream:
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

    def run_loader(
        self,
        loader,
        epochs: int = 1,
    ) -> Dict:
        """Train from a DataLoader (KAB2Dataset or any map-style dataset)."""
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
        """
        Collate items into a batch dict.

        Items can be:
          - (moves_b, moves_w, info_b, info_w) tuples (from stream_to_tensors)
          - KAB2Sample dicts (from KAB2StreamDataset)
        """
        # Detect by shape: if first element is a dict, treat as KAB2Sample
        if items and isinstance(items[0], dict) and 'game_id' in items[0]:
            return kab2_collate(items)

        # Otherwise treat as stream_to_tensors output
        samples = []
        for moves_b, moves_w, info_b, info_w in items:
            samples.append(kab2_make_sample(
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
        return kab2_collate(samples)


# ─── Inference Workflow ───────────────────────────────────────────────────────

class InferenceWorkflow:
    """
    Batch rank inference for SGF files or raw SGF strings.

    Returns per-game rank distributions without training.
    """

    def __init__(
        self,
        model: nn.Module,
        engine: 'KataGoEngine',
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
    ) -> List[RankResult]:
        """Rank players in a list of SGF file paths."""
        results = []
        for x_b, x_w, info_b, info_w in self.engine.stream_to_tensors(
            sgf_paths=sgfs, mode=mode, min_moves=min_moves
        ):
            result = self._infer_one(x_b, x_w, info_b, info_w)
            results.append(result)
        return results

    def rank_strings(
        self,
        sgf_strings: List[str],
        mode: str = 'lite',
        min_moves: int = 10,
    ) -> List[RankResult]:
        """Rank players from in-memory SGF strings."""
        results = []
        for x_b, x_w, info_b, info_w in self.engine.stream_to_tensors(
            sgf_strings=sgf_strings, mode=mode, min_moves=min_moves
        ):
            result = self._infer_one(x_b, x_w, info_b, info_w)
            results.append(result)
        return results

    @torch.no_grad()
    def _infer_one(self, x_b, x_w, info_b, info_w) -> RankResult:
        sample = kab2_make_sample(
            moves_b = x_b,
            moves_w = x_w,
            game_id = info_b.get('game_id', ''),
        )
        out = self.model(sample['x'].to(self.device), xlens=[sample['seq_len']])

        def _rank_probs(logits):
            return torch.softmax(logits, dim=-1).squeeze(0).cpu()

        return RankResult(
            game_id      = sample['game_id'],
            b_log_prior  = out.get('b_log_prior', out.get('b_rating', torch.tensor(0.))).item(),
            w_log_prior  = out.get('w_log_prior', out.get('w_rating', torch.tensor(0.))).item(),
            b_rank_probs = _rank_probs(out['b_rank_logits']) if 'b_rank_logits' in out else None,
            w_rank_probs = _rank_probs(out['w_rank_logits']) if 'w_rank_logits' in out else None,
            b_rank_top   = int(out['b_rank_logits'].argmax(-1)) if 'b_rank_logits' in out else -1,
            w_rank_top   = int(out['w_rank_logits'].argmax(-1)) if 'w_rank_logits' in out else -1,
        )
