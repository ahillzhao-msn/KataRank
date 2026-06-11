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
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from katarank.schema import (
    KAB2Output, KAB2Sample, MoveRecord, ReviewOutput,
    kab2_make_sample, kab2_collate,
)


# ─── Inference result type ────────────────────────────────────────────────────

class RankResult(TypedDict):
    """Per-game inference output from InferenceWorkflow."""
    game_id:      str
    b_rating:     float            # continuous strength signal for Black
    w_rating:     float            # continuous strength signal for White
    b_rank_probs: Optional[torch.Tensor]  # (29,) probabilities over rank classes
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
            if not batch['xlens']:
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
            if batch['xlens']:
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
                if not batch['xlens']:
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

        # KataRankLoss convention: loss_fn(predictions, targets) -> dict with 'total'
        targets = {
            k: batch[k].to(self.device)
            for k in ('target_b', 'target_w', 'rank_b', 'rank_w',
                      'human_lp_b', 'human_lp_w')
        }
        losses = self.loss_fn(out, targets)
        loss = losses['total'] if isinstance(losses, dict) else losses

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

        # Otherwise treat as stream_to_tensors output (torch tensors)
        samples = []
        for moves_b, moves_w, info_b, info_w in items:
            if isinstance(moves_b, torch.Tensor):
                moves_b = moves_b.cpu().numpy()
            if isinstance(moves_w, torch.Tensor):
                moves_w = moves_w.cpu().numpy()
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
            moves_b = x_b.cpu().numpy(),
            moves_w = x_w.cpu().numpy(),
            game_id = info_b.get('game_id', ''),
        )
        out = self.model(sample['x'].to(self.device), xlens=[sample['seq_len']])

        # KataRankModel.forward keys: b_rating / w_rating / rank_probs_b / rank_probs_w
        # (rank_probs_* are already probabilities — no softmax here)
        probs_b = out.get('rank_probs_b')
        probs_w = out.get('rank_probs_w')

        return RankResult(
            game_id      = sample['game_id'],
            b_rating     = out['b_rating'].item(),
            w_rating     = out['w_rating'].item(),
            b_rank_probs = probs_b.squeeze(0).cpu() if probs_b is not None else None,
            w_rank_probs = probs_w.squeeze(0).cpu() if probs_w is not None else None,
            b_rank_top   = int(probs_b.argmax(-1)) if probs_b is not None else -1,
            w_rank_top   = int(probs_w.argmax(-1)) if probs_w is not None else -1,
        )


# ─── RankResult / engine stats → KAB2Output ──────────────────────────────────
#
# Shared by the CLI (katarank-infer) and the REST API: both are thin shells
# over these functions.

def result_to_output(r: RankResult) -> KAB2Output:
    """Convert an InferenceWorkflow RankResult into a KAB2Output.

    Confidence = max rank-class probability (model's certainty about the
    predicted rank), 0.0 when the model produced no rank distribution.
    """
    def _conf(probs):
        return float(probs.max()) if probs is not None else 0.0

    return KAB2Output(
        game_id      = r['game_id'],
        metadata     = {'source': 'model'},
        b_rating     = r['b_rating'],
        w_rating     = r['w_rating'],
        b_rank       = r['b_rank_top'],
        w_rank       = r['w_rank_top'],
        b_confidence = _conf(r['b_rank_probs']),
        w_confidence = _conf(r['w_rank_probs']),
        b_rank_probs = r['b_rank_probs'].tolist() if r['b_rank_probs'] is not None else None,
        w_rank_probs = r['w_rank_probs'].tolist() if r['w_rank_probs'] is not None else None,
    )


def engine_stats_outputs(stream: Iterable) -> List[KAB2Output]:
    """Build KAB2Outputs from raw engine statistics, paired by game_id.

    Used when no KataRankModel checkpoint is loaded: b_rating/w_rating fall
    back to meanLogPrior; b_rank/w_rank to the HumanSL annotation (-1 if the
    engine ran without -human-model). Confidence here is a heuristic in
    [0, 1] derived from meanLogPrior magnitude — NOT comparable to the
    model-based confidence in result_to_output().
    """
    games: Dict[str, KAB2Output] = {}
    order: List[str] = []
    for side, _moves, info in stream:
        gid = info.get('game_id') or f'game_{len(order):04d}'
        if gid not in games:
            games[gid] = KAB2Output(
                game_id=gid,
                metadata={'source': 'engine'},
                b_rating=0.0, w_rating=0.0,
                b_rank=-1, w_rank=-1,
                b_confidence=0.0, w_confidence=0.0,
                b_rank_probs=None, w_rank_probs=None,
            )
            order.append(gid)
        entry = games[gid]
        confidence = max(0.0, min(1.0, 1.0 - abs(info['mean_log_prior']) / 10.0))
        if side == 'B':
            entry['b_rating']     = float(info['mean_log_prior'])
            entry['b_rank']       = info['human_rank_idx']
            entry['b_confidence'] = confidence
        elif side == 'W':
            entry['w_rating']     = float(info['mean_log_prior'])
            entry['w_rank']       = info['human_rank_idx']
            entry['w_confidence'] = confidence
    return [games[g] for g in order]


def run_rank_files(
    engine, inf_workflow: Optional[InferenceWorkflow],
    paths: List[str], mode: str = 'lite', min_moves: int = 10,
) -> List[KAB2Output]:
    """Rank SGF files: model inference if a workflow is given, engine stats otherwise.

    KAB2Output.metadata is populated from the SGF headers (players, date,
    rules, komi, result, ...), matched by game id = filename stem.
    """
    if inf_workflow is not None:
        rr = inf_workflow.rank_files(paths, mode=mode, min_moves=min_moves)
        outputs = [result_to_output(r) for r in rr]
    else:
        outputs = engine_stats_outputs(
            engine.stream_games(sgf_paths=paths, mode=mode, min_moves=min_moves)
        )
    _attach_metadata_from_files(outputs, paths)
    return outputs


def run_rank_strings(
    engine, inf_workflow: Optional[InferenceWorkflow],
    strings: List[str], mode: str = 'lite', min_moves: int = 10,
) -> List[KAB2Output]:
    """Rank SGF strings: model inference if a workflow is given, engine stats otherwise.

    KAB2Output.metadata is populated from the SGF headers; string inputs are
    matched back via the deterministic '_string_NNNNNN' game ids.
    """
    if inf_workflow is not None:
        rr = inf_workflow.rank_strings(strings, mode=mode, min_moves=min_moves)
        outputs = [result_to_output(r) for r in rr]
    else:
        outputs = engine_stats_outputs(
            engine.stream_games(sgf_strings=strings, mode=mode, min_moves=min_moves)
        )
    _attach_metadata_from_strings(outputs, strings)
    return outputs


# ─── Per-move review (docs/REVIEW_API_DESIGN.md) ─────────────────────────────

def _move_records(moves_b: np.ndarray, moves_w: np.ndarray) -> List[MoveRecord]:
    """Convert raw KAB2 move matrices into mover-perspective MoveRecords.

    Scalar layout (white's perspective, batch_analysis.cpp appendMoveRecord):
      [0] whiteWinProb [1] whiteLossProb [2] whiteNoResultProb
      [3] whiteScoreMean/50 [4] shorttermScoreError/10
      [5] policyPrior [6] policyRank/361 [7] isWhite
      [8] winDelta [9] scoreDelta/50

    Move numbers assume strict B/W alternation, Black first — the same
    assumption DualViewSetTransformer uses for causal masking.
    """
    records: List[MoveRecord] = []
    for stream, color in ((moves_b, 'B'), (moves_w, 'W')):
        sign = 1.0 if color == 'W' else -1.0
        for i, row in enumerate(stream):
            records.append(MoveRecord(
                move_no      = 2 * i + (1 if color == 'B' else 2),
                color        = color,
                winrate      = float(row[0] if color == 'W' else row[1]),
                score_lead   = float(sign * row[3] * 50.0),
                score_stdev  = float(row[4] * 10.0),
                policy_prior = float(row[5]),
                policy_rank  = int(round(float(row[6]) * 361.0)),
                win_delta    = float(sign * row[8]),
                score_delta  = float(sign * row[9] * 50.0),
            ))
    records.sort(key=lambda r: r['move_no'])
    return records


def _review_from_stream(
    stream: Iterable,
    inf_workflow: Optional[InferenceWorkflow],
) -> List[ReviewOutput]:
    """Build ReviewOutputs from one pass over a (side, moves, info) stream.

    The whole-game verdict reuses the /rank logic: model inference when a
    workflow is given, meanLogPrior heuristics otherwise — review never
    costs a second katago run.
    """
    games: Dict[str, Dict] = {}
    order: List[str] = []
    for side, moves, info in stream:
        gid = info.get('game_id') or f'game_{len(order):04d}'
        if gid not in games:
            games[gid] = {}
            order.append(gid)
        games[gid][side] = (moves, info)

    outputs: List[ReviewOutput] = []
    for gid in order:
        entry = games[gid]
        dim = next(m.shape[1] for m, _ in entry.values())
        empty = np.zeros((0, dim), dtype=np.float32)
        moves_b, info_b = entry.get('B', (empty, {}))
        moves_w, info_w = entry.get('W', (empty, {}))
        recs = _move_records(moves_b, moves_w)

        if inf_workflow is not None:
            rr = inf_workflow._infer_one(
                torch.from_numpy(np.ascontiguousarray(moves_b)),
                torch.from_numpy(np.ascontiguousarray(moves_w)),
                {**info_b, 'game_id': gid}, info_w,
            )
            base = result_to_output(rr)
        else:
            def _frames():
                for side_, (m, i) in entry.items():
                    yield side_, m, {**i, 'game_id': gid}
            base = engine_stats_outputs(_frames())[0]

        outputs.append(ReviewOutput(**base, moves=recs))
    return outputs


def run_review_files(
    engine, inf_workflow: Optional[InferenceWorkflow],
    paths: List[str], mode: str = 'lite', min_moves: int = 10,
) -> List[ReviewOutput]:
    """Review SGF files: whole-game verdict + per-move records, one engine pass."""
    outputs = _review_from_stream(
        engine.stream_games(sgf_paths=paths, mode=mode, min_moves=min_moves),
        inf_workflow,
    )
    _attach_metadata_from_files(outputs, paths)
    return outputs


def run_review_strings(
    engine, inf_workflow: Optional[InferenceWorkflow],
    strings: List[str], mode: str = 'lite', min_moves: int = 10,
) -> List[ReviewOutput]:
    """Review SGF strings: whole-game verdict + per-move records, one engine pass."""
    outputs = _review_from_stream(
        engine.stream_games(sgf_strings=strings, mode=mode, min_moves=min_moves),
        inf_workflow,
    )
    _attach_metadata_from_strings(outputs, strings)
    return outputs


def _attach_metadata_from_files(outputs: List[KAB2Output], paths: List[str]):
    from pathlib import Path
    from katarank.sgf_meta import read_sgf_metadata
    by_stem = {Path(p).stem: p for p in paths}
    for o in outputs:
        p = by_stem.get(o['game_id'])
        if p:
            o['metadata'].update(read_sgf_metadata(p))


def _attach_metadata_from_strings(outputs: List[KAB2Output], strings: List[str]):
    import re
    from katarank.sgf_meta import parse_sgf_metadata
    for o in outputs:
        m = re.fullmatch(r'_string_(\d+)', o['game_id'])
        if m:
            idx = int(m.group(1))
            if idx < len(strings):
                o['metadata'].update(parse_sgf_metadata(strings[idx]))
