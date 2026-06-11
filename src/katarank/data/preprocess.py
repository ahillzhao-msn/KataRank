"""
KataRank — KAB2 binary format reader.

KAB2 layout
-----------
Offset  Size  Field
     0     4  magic b'KAB2'
     4     4  numMoves         (int32)
     8     4  scalarDim        (int32, always 10)
    12     4  trunkDim         (int32, 0 if -no-trunk was used)
    16     4  pickDim          (int32)
    20     4  nnXLen           (int32)
    24     4  nnYLen           (int32)
    28     4  flags            (int32, bit0=zlib compressed)
    32    64  PlayerSummary    (16×float32)
  ----   ---
    96        payload          (numMoves × moveDim float32, optionally zlib'd)

PlayerSummary notable indices
    [2]  meanLogPrior      — primary training target
    [10] humanRankIdx      — 0..28 (20k..9d), -1 if not computed
    [11] humanLogPrior     — HumanSL confidence weight

Scalar layout per move (10 fields)
    [0] whiteWinProb
    [1] whiteLossProb
    [2] whiteNoResultProb
    [3] whiteScoreMean/50
    [4] shorttermScoreError/10
    [5] policyPrior
    [6] policyRank/361
    [7] isWhite  (0=Black, 1=White)
    [8] winDelta (deferred — 0.0 in file)
    [9] scoreDelta/50 (deferred — 0.0 in file)
"""

import struct
import zlib
from typing import Dict, Optional, Tuple

import numpy as np


_MAGIC = b'KAB2'
_HEADER_SIZE = 96

# PlayerSummary field indices
KAB2_SUM_MEAN_LP     = 2
KAB2_SUM_HUMAN_RANK  = 10
KAB2_SUM_HUMAN_LP    = 11

# Scalar field indices
KAB2_SCALAR_WIN_PROB   = 0
KAB2_SCALAR_LOSS_PROB  = 1
KAB2_SCALAR_SCORE      = 3
KAB2_SCALAR_IS_WHITE   = 7
KAB2_SCALAR_WIN_DELTA  = 8


def _is_combined(raw: bytes) -> bool:
    """Detect combined format: [4B B_size][B KAB2][4B W_size][W KAB2].

    The KAB2 magic sits at offset 4 (B present) or offset 8 (B empty:
    the first 4 bytes are a zero size, the next 4 the W size).
    """
    if len(raw) > 8 and raw[4:8] == _MAGIC:
        return True
    if len(raw) > 12 and raw[:4] == b'\x00\x00\x00\x00' and raw[8:12] == _MAGIC:
        return True
    return False


def read_kab2(path: str) -> Tuple[np.ndarray, Dict]:
    """Read a KAB2 file (single or combined) and return (moves, info).

    For combined .npz files (B+W in one file), returns Black player data,
    falling back to White if the Black side is empty.
    Use read_kab2_combined() to get both players.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    if _is_combined(raw):
        b_moves, w_moves, b_info, w_info = _parse_combined(raw, path)
        if b_moves is not None:
            return b_moves, b_info
        if w_moves is not None:
            return w_moves, w_info
        raise ValueError(f"Combined KAB2 file has no player data: {path}")
    # Old single format
    return _parse_kab2(raw)


def probe_kab2_dim(path: str) -> int:
    """Read header and return moveDim without loading move data.

    Handles both old single and new combined .npz formats.
    """
    with open(path, 'rb') as f:
        header = f.read(_HEADER_SIZE + 12)  # enough for combined header too
    # Combined: KAB2 magic at offset 4 (B present) or 8 (B empty, W present)
    for off in (4, 8):
        if len(header) >= off + 20 and header[off: off + 4] == _MAGIC:
            scalar_dim = struct.unpack_from('<i', header, off + 8)[0]
            trunk_dim  = struct.unpack_from('<i', header, off + 12)[0]
            return scalar_dim + 2 * trunk_dim
    # Old single format: header starts at byte 0
    if header[:4] != _MAGIC:
        raise ValueError(f"Not a KAB2 file: {path}")
    scalar_dim = struct.unpack_from('<i', header, 8)[0]
    trunk_dim  = struct.unpack_from('<i', header, 12)[0]
    return scalar_dim + 2 * trunk_dim


def _parse_kab2(raw: bytes) -> Tuple[np.ndarray, Dict]:
    if raw[:4] != _MAGIC:
        raise ValueError(f"Not a KAB2 buffer (magic={raw[:4]!r})")

    num_moves  = struct.unpack_from('<i', raw, 4)[0]
    scalar_dim = struct.unpack_from('<i', raw, 8)[0]
    trunk_dim  = struct.unpack_from('<i', raw, 12)[0]
    flags      = struct.unpack_from('<i', raw, 28)[0]
    summary    = struct.unpack_from('<16f', raw, 32)

    payload = raw[_HEADER_SIZE:]
    if flags & 1:
        clen = struct.unpack_from('<I', payload, 0)[0]
        payload = zlib.decompress(payload[4: 4 + clen])

    move_dim = scalar_dim + 2 * trunk_dim
    moves = np.frombuffer(payload, dtype=np.float32).reshape(num_moves, move_dim)
    moves = np.ascontiguousarray(moves)

    human_rank = int(summary[KAB2_SUM_HUMAN_RANK]) if summary[KAB2_SUM_HUMAN_RANK] >= 0 else -1

    info = {
        'num_moves':       num_moves,
        'scalar_dim':      scalar_dim,
        'trunk_dim':       trunk_dim,
        'input_dim':       move_dim,
        'compressed':      bool(flags & 1),
        'summary':         summary,
        'mean_log_prior':  float(summary[KAB2_SUM_MEAN_LP]),
        'human_rank_idx':  human_rank,
        'human_log_prior': float(summary[KAB2_SUM_HUMAN_LP]),
    }
    return moves, info


# ─── Combined KAB2 pair reader ───────────────────────────────────────────────

def read_kab2_combined(
    path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict, Dict]:
    """
    Read a combined .npz file containing both Black and White KAB2 payloads.

    Format: [4B B_payload_size][B_KAB2_payload][4B W_payload_size][W_KAB2_payload]
    Returns (b_moves, w_moves, b_info, w_info); a missing side is (None, {}).
    """
    with open(path, 'rb') as f:
        data = f.read()
    return _parse_combined(data, path)


def _parse_combined(
    data: bytes, path: str = '<buffer>',
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict, Dict]:
    if len(data) < 8:
        raise ValueError(f"Combined KAB2 file too short: {path}")

    b_sz = struct.unpack_from('<I', data, 0)[0]
    if 4 + b_sz + 4 > len(data):
        raise ValueError(f"Combined KAB2 file truncated (B section): {path}")
    w_sz = struct.unpack_from('<I', data, 4 + b_sz)[0]
    if 8 + b_sz + w_sz > len(data):
        raise ValueError(f"Combined KAB2 file truncated (W section): {path}")

    b_moves = w_moves = None
    b_info  = w_info  = None

    if b_sz > 0:
        b_moves, b_info = _parse_kab2(data[4: 4 + b_sz])
    if w_sz > 0:
        w_moves, w_info = _parse_kab2(data[8 + b_sz: 8 + b_sz + w_sz])

    return b_moves, w_moves, b_info or {}, w_info or {}
