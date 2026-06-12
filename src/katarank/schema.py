"""
KataRank — Data Schema
=======================
Defines the canonical data contract for the project.

- BaseKAB2Dataset: abstract base for all data sources
- KAB2Sample: per-game sample format
- KAB2Batch: collated batch format
- kab2_make_sample / kab2_collate: sample & batch construction
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict, Union
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import torch


# ─── Canonical sample type ────────────────────────────────────────────────────

class KAB2Sample(TypedDict):
    """One paired game sample — both players' moves concatenated.

    Tensor layout: x is (N_b + N_w, input_dim) float32.
    First N_b rows: Black moves (scalar[7] == 0).
    Last  N_w rows: White moves (scalar[7] == 1).
    """
    x:          torch.Tensor   # (N_b + N_w, input_dim) float32
    seq_len:    int
    target_b:   torch.Tensor   # meanLogPrior for Black (rating signal)
    target_w:   torch.Tensor   # meanLogPrior for White
    rank_b:     torch.Tensor   # humanRankIdx (0-28 or -1)
    rank_w:     torch.Tensor
    human_lp_b: torch.Tensor   # HumanSL confidence weight
    human_lp_w: torch.Tensor
    game_id:    str


class KAB2Batch(TypedDict):
    """Collated batch for model.forward().

    Produced by kab2_collate(). Variable-length games are packed into
    a flat tensor; xlens tracks per-game sequence lengths.
    """
    x:          torch.Tensor   # (sum_N, input_dim)
    xlens:      List[int]      # per-game lengths, len = batch_size
    target_b:   torch.Tensor   # (batch_size,)
    target_w:   torch.Tensor
    rank_b:     torch.Tensor
    rank_w:     torch.Tensor
    human_lp_b: torch.Tensor
    human_lp_w: torch.Tensor
    game_ids:   List[str]


# ─── Abstract base ───────────────────────────────────────────────────────────

class BaseKAB2Dataset(ABC):
    """
    Abstract base for all KAB2 data sources.

    Properties:
        input_dim: Number of features per move (10 for lite mode,
                   10 + 2*trunkCh for full mode).

    Yields (via __getitem__ or __iter__):
        KAB2Sample — one game with both players' moves concatenated.

    Subclasses must implement:
        input_dim (property)
        __len__ (int, number of games)
    """

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Feature dimensionality per move."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of games in this dataset."""
        ...

    # Optional: allow __getitem__ for map-style access
    @abstractmethod
    def __getitem__(self, idx: int) -> KAB2Sample:
        ...


def kab2_make_sample(
    moves_b: 'np.ndarray',
    moves_w: 'np.ndarray',
    game_id: str,
    target_b: float = 0.0,
    target_w: float = 0.0,
    rank_b: int = -1,
    rank_w: int = -1,
    human_lp_b: float = 0.0,
    human_lp_w: float = 0.0,
) -> KAB2Sample:
    """Build a KAB2Sample from numpy move arrays (Black first, then White)."""
    import numpy as np
    x = np.concatenate([moves_b, moves_w], axis=0)
    return KAB2Sample(
        x          = torch.from_numpy(x),
        seq_len    = len(x),
        target_b   = torch.tensor(target_b,   dtype=torch.float32),
        target_w   = torch.tensor(target_w,   dtype=torch.float32),
        rank_b     = torch.tensor(rank_b,     dtype=torch.long),
        rank_w     = torch.tensor(rank_w,     dtype=torch.long),
        human_lp_b = torch.tensor(human_lp_b, dtype=torch.float32),
        human_lp_w = torch.tensor(human_lp_w, dtype=torch.float32),
        game_id    = game_id,
    )


# ─── Collate ──────────────────────────────────────────────────────────────────

def kab2_collate(batch: List[Union[KAB2Sample, Dict]]) -> KAB2Batch:
    """
    Pack variable-length game sequences into a flat tensor + xlens list.

    Filters out empty samples (games with too few moves).
    """
    batch = [item for item in batch if item['seq_len'] > 0]
    if not batch:
        return KAB2Batch(
            x=torch.empty(0), xlens=[], target_b=torch.empty(0),
            target_w=torch.empty(0), rank_b=torch.empty(0),
            rank_w=torch.empty(0), human_lp_b=torch.empty(0),
            human_lp_w=torch.empty(0), game_ids=[],
        )

    x_list  = [item['x'] for item in batch]
    xlens   = [item['seq_len'] for item in batch]

    return KAB2Batch(
        x          = torch.cat(x_list, dim=0),
        xlens      = xlens,
        target_b   = torch.stack([item['target_b']   for item in batch]),
        target_w   = torch.stack([item['target_w']   for item in batch]),
        rank_b     = torch.stack([item['rank_b']     for item in batch]),
        rank_w     = torch.stack([item['rank_w']     for item in batch]),
        human_lp_b = torch.stack([item['human_lp_b'] for item in batch]),
        human_lp_w = torch.stack([item['human_lp_w'] for item in batch]),
        game_ids   = [item['game_id'] for item in batch],
    )


# ─── Model inference output ──────────────────────────────────────────────────

class KAB2Output(TypedDict):
    """Model inference output — one per game, symmetric with KAB2Sample.

    Core output: per-player rating + rank + confidence.
    Optional: full rank probability distributions (29-dim).

    Confidence semantics (always in [0, 1]; check metadata['source']):
      source='model'  — max rank-class probability from the trained
                        KataRankModel (its certainty about the argmax rank)
      source='engine' — heuristic 1 - |meanLogPrior|/10, clamped; a rough
                        signal quality proxy, NOT comparable to model
                        confidence

    metadata carries SGF header fields when available: player_black,
    player_white, black_rank, white_rank, date, rules, komi, result,
    board_size, event, handicap — plus 'source' as above.
    """
    game_id:    str
    metadata:   Dict          # game metadata (date, rules, komi, player names, ...)

    b_rating:   float
    w_rating:   float
    b_rank:     int           # 0=20k … 28=9d
    w_rank:     int
    b_confidence: float       # reliability (game quality, move count, ...)
    w_confidence: float

    b_rank_probs: Optional[List[float]]  # 29-dim probability distribution
    w_rank_probs: Optional[List[float]]


class MoveRecord(TypedDict):
    """Per-move review record derived from KAB2 move scalars.

    All evaluative fields are in the MOVER's perspective (sign-flipped for
    Black from KAB2's white-perspective raw scalars). Layout and conversion
    rules: docs/REVIEW_API_DESIGN.md §2.

    Move numbers assume strict B/W alternation with Black first (same
    assumption as DualViewSetTransformer's causal masks); handicap games
    are off by their handicap offset.
    """
    move_no:      int      # 1-based global move number
    color:        str      # 'B' | 'W'
    winrate:      float    # mover win probability at this position
    score_lead:   float    # mover expected score lead (points)
    score_stdev:  float    # shortterm score error — position complexity proxy
    policy_prior: float    # policy prior of the move actually played
    policy_rank:  int      # rank of played move in policy (0 = engine's top)
    win_delta:    float    # mover winrate change caused by this move
    score_delta:  float    # mover score change caused by this move (points)
    ownership:    NotRequired[Optional[List[float]]]  # 361 floats, stream mode only


class ReviewOutput(KAB2Output):
    """Game review output: KAB2Output (whole-game verdict) + per-move records.

    Returned by /review/* endpoints. The 'moves' list is ascending by
    move_no. Reserved for the future SAE review workflow: each move object
    will additionally carry 'features' (see docs/SAE_DESIGN.md).
    """
    moves: List[MoveRecord]


RANK_NAMES = [
    '20k','19k','18k','17k','16k','15k','14k','13k','12k','11k',
    '10k','9k','8k','7k','6k','5k','4k','3k','2k','1k',
    '1d','2d','3d','4d','5d','6d','7d','8d','9d',
]


def rank_idx_to_str(idx: int) -> str:
    """Convert rank index 0..28 → '20k' … '9d'. Returns '?' for invalid."""
    return RANK_NAMES[idx] if 0 <= idx < len(RANK_NAMES) else '?'


# ─── Training output ─────────────────────────────────────────────────────────

class TrainingReport(TypedDict):
    """Training run summary — the canonical training output, symmetric with
    KAB2Output on the inference side. Written next to the checkpoints.

    final_metrics keys (computed on the validation split):
        rank_mae       mean |predicted - label| in rank steps (stones)
        rank_acc       exact rank accuracy
        rank_acc_pm1   accuracy within ±1 rank
        rating_corr    Pearson r between predicted rating and meanLogPrior
        n_rank_labeled number of player-sides with HumanSL labels
    """
    version:          str           # report schema version
    created_at:       str           # ISO-8601 timestamp
    model_config:     Dict          # KataRankModel config snapshot
    training_config:  Dict          # training section snapshot
    data:             Dict          # games per split, label coverage, input_dim
    epochs_trained:   int
    early_stopped:    bool
    best_epoch:       int
    best_val_loss:    float
    train_loss:       List[float]   # per epoch
    val_loss:         List[float]   # per epoch
    final_metrics:    Dict
    ordinal_thresholds_b: List[float]   # learned rank boundaries (strength axis)
    ordinal_thresholds_w: List[float]
    elapsed_seconds:  float


def save_training_report(report: TrainingReport, path: str):
    """Save a TrainingReport as JSON."""
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(report), f, ensure_ascii=False, indent=2)


def load_training_report(path: str) -> TrainingReport:
    """Load a TrainingReport from JSON."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return TrainingReport(**json.load(f))


# ─── Serialization ───────────────────────────────────────────────────────────

def output_to_json(out: KAB2Output, indent: int = 2) -> str:
    """Serialize a single KAB2Output to JSON string."""
    import json
    d = dict(out)
    # Convert rank probs from Optional[List] — handled naturally by json
    return json.dumps(d, indent=indent, ensure_ascii=False)


def output_from_json(s: str) -> KAB2Output:
    """Deserialize a KAB2Output from JSON string."""
    import json
    d = json.loads(s)
    return KAB2Output(**d)


def save_output(out: KAB2Output, path: str):
    """Save KAB2Output to file. Auto-detect format by extension."""
    if path.endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(output_to_json(out))
    elif path.endswith('.json.gz'):
        import gzip
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(output_to_json(out))
    else:
        raise ValueError(f"Unsupported extension: {path} (use .json or .json.gz)")


def load_output(path: str) -> KAB2Output:
    """Load KAB2Output from file. Auto-detect format by extension."""
    if path.endswith('.json.gz'):
        import gzip
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return output_from_json(f.read())
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return output_from_json(f.read())
    else:
        raise ValueError(f"Unsupported extension: {path} (use .json or .json.gz)")


def save_outputs_batch(outputs: List[KAB2Output], path: str):
    """Save multiple KAB2Outputs.

    .jsonl / .jsonl.gz — one JSON object per line (stream friendly)
    .json  / .json.gz  — single JSON array (archive friendly)
    """
    import json as _json

    def _open(p):
        if p.endswith('.gz'):
            import gzip
            return gzip.open(p, 'wt', encoding='utf-8')
        return open(p, 'w', encoding='utf-8')

    if path.endswith('.jsonl') or path.endswith('.jsonl.gz'):
        with _open(path) as f:
            for o in outputs:
                f.write(_json.dumps(dict(o), ensure_ascii=False) + '\n')
    elif path.endswith('.json') or path.endswith('.json.gz'):
        with _open(path) as f:
            _json.dump([dict(o) for o in outputs], f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(
            f"Unsupported extension: {path} (use .json[.gz] or .jsonl[.gz])"
        )


def load_outputs_batch(path: str) -> List[KAB2Output]:
    """Load multiple KAB2Outputs from JSONL (.jsonl or .jsonl.gz)."""
    import json as _json
    outputs = []
    if path.endswith('.jsonl.gz'):
        import gzip
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    outputs.append(KAB2Output(**_json.loads(line)))
    elif path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    outputs.append(KAB2Output(**_json.loads(line)))
    else:
        raise ValueError(f"Unsupported extension: {path} (use .jsonl or .jsonl.gz)")
    return outputs
