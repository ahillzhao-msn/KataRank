# KataRank KAB2 Migration Report

**Date**: 2026-06-09  
**Scope**: Migrate `katarank` Python training pipeline from legacy `.npy`/`.zip`/`KABN` format to the new `KAB2` binary format produced by `batch_analysis`.

---

## 1. Executive Summary

The model architecture (`dual_view.py`, `set_transformer.py`, `multi_task.py`) requires **zero changes** — `input_proj = nn.Linear(input_dim, hidden_dim)` already handles arbitrary input dimensions. All changes are in the **data loading layer** and **training configuration**:

| Layer | Status | Action |
|---|---|---|
| `model/set_transformer.py` | Compatible | No change |
| `model/dual_view.py` | Compatible | No change |
| `model/multi_task.py` | Compatible | No change |
| `model/losses.py` | Compatible | No change |
| `data/katago_native/dataset.py` | BROKEN | Replace with KAB2 dataset |
| `data/preprocess.py` | Partial | Add KAB2 loader function |
| `train/train_kata_native.py` | BROKEN | Update feature_dim + labels |
| `train/config_kata_native.yaml` | BROKEN | Update input_dim + paths |

---

## 2. KAB2 Binary Format

Each game produces two files: `game_{id:016X}_B.npz` (Black) and `game_{id:016X}_W.npz` (White).

### 2.1 File Layout

```
[NPZHeader — 96 bytes]
[Move records — N × moveDim × sizeof(float32)]
```

**NPZHeader** (packed, no alignment):
```c
struct NPZHeader {   // 96 bytes total
    char    magic[4];       // "KAB2"
    int32_t numMoves;       // N moves for this player
    int32_t scalarDim;      // always 10
    int32_t trunkDim;       // pick channels (= trunkCh, e.g. 512 or 384)
    int32_t pickDim;        // avgTrunk channels (= trunkCh, same value)
    int32_t nnXLen;         // board width (19)
    int32_t nnYLen;         // board height (19)
    int32_t flags;          // bit0 = zlib compressed
    PlayerSummary summary;  // 16 × float32 = 64 bytes
};
```

**PlayerSummary** (16 float32 = 64 bytes):
```
[0]  accuracy1       — top-1 accuracy (fraction of moves matching KataGo #1)
[1]  accuracy3       — top-3 accuracy
[2]  meanLogPrior    — mean(log(policy[actualMove])), range ~ [-5.9, -0.5]
[3]  meanWinRate     — mean win rate during player's moves
[4]  meanScoreLead   — mean score lead
[5]  meanComplexity  — mean 1-max(policy)
[6]  scoreVariance   — Welford variance of score lead
[7]  approxScoreDrop — approx score drop (KataGo eval quality)
[8]  meanWinDelta    — mean win rate change per move
[9]  meanScoreDelta  — mean score lead change per move
[10] humanRankIdx    — 0-28 (rank_20k through rank_9d), -1 if not computed
[11] humanLogPrior   — mean log-likelihood under best-match HumanSL profile
[12..15] reserved
```

**Move record** (`moveDim = scalarDim + trunkDim + pickDim`):
```
[0..9]           scalars       (10 float32)
[10..10+trunkCh-1]  pick       (trunkCh float32) — trunk activation at chosen move position
[10+trunkCh..10+2*trunkCh-1]  avgTrunk  (trunkCh float32) — spatial mean of trunk
```

**Scalar layout** (indices 0–9):
```
[0] winRate          — KataGo winrate estimate (0-1)
[1] scoreLead        — score lead (positive = current player winning)
[2] complexity       — 1 - max(policy), uncertainty measure
[3] policyEntropy    — entropy of policy distribution
[4] priorProb        — policy prob of the actual move chosen
[5] winDelta         — change in win rate from previous position
[6] scoreDelta       — change in score lead from previous position
[7] isWhite          — 0 for Black, 1 for White  ← player split key
[8] turnNumber       — half-move number
[9] boardArea        — board area (361 for 19×19)
```

### 2.2 Compression

If `flags & 1 == 1`, the move records (everything after the 96-byte header) are zlib-compressed. The compressed data is preceded by a 4-byte little-endian int giving the compressed size.

### 2.3 Reading KAB2 in Python

```python
import struct, zlib
import numpy as np

def read_kab2(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'KAB2', f"Bad magic: {magic}"
        n, sd, td, pd, nx, ny, flags = struct.unpack('<7i', f.read(28))
        summary = np.frombuffer(f.read(64), dtype=np.float32).copy()
        raw = f.read()
    if flags & 1:
        cl = struct.unpack('<i', raw[:4])[0]
        raw = zlib.decompress(raw[4:4+cl])
    move_dim = sd + td + pd
    moves = np.frombuffer(raw, dtype=np.float32).reshape(n, move_dim).copy()
    return moves, summary, {'n': n, 'scalar_dim': sd, 'trunk_dim': td, 'nx': nx, 'ny': ny}
```

---

## 3. Current Codebase: What Breaks

### 3.1 `data/katago_native/dataset.py` — `KataNativeDataset`

**Breaks because:**
- Calls `load_pick_features(feature_dir, sgf_path, player)` which looks for `.npy` / `.zip` files (go-strength-model format)
- New files are `game_{id:016X}_B.npz` / `_W.npz` (KAB2 format, different extension and layout)
- No KAB2 parser exists in the data layer

**What it does correctly (to preserve):**
- Variable-length sequence handling via `pack_batch` + `xlens`
- `KataNativeDataLoader` collate function (can be reused as-is)
- `compute_elo_metrics` utility

### 3.2 `train/train_kata_native.py`

**Breaks because:**
```python
feature_dim = 256  # HARDCODED — was pick_dim for old 256-channel model
```
New: `feature_dim = scalarDim + 2 * trunkCh` (= 10 + 2×512 = 1034 for katago b18)

**Labels break too:**
- Currently reads `BlackRating`/`WhiteRating` from CSV (Katago weight Elo)
- New: labels come from `summary[2]` (`meanLogPrior`) embedded in KAB2 header — no separate CSV needed

**Model instantiation breaks:**
- Current: `enable_score=True` (single rating) with `per_player=True` sampling
- Target: `enable_dual_rating=True` (B+W pair), game-level input (both players concatenated)

### 3.3 `train/config_kata_native.yaml`

```yaml
model:
  input_dim: 256  # WRONG — should be 1034 (or dynamic)
  feature_name: "pick"  # WRONG — now we load full KAB2 (scalars+pick+trunk)
```

---

## 4. Architecture Compatibility Assessment

### 4.1 `model/dual_view.py` — DualViewSetTransformer

**Fully compatible.** Key design that makes this work:

```python
self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
```

With `input_dim=1034` (or any dynamic value read from header), this projects the full `(scalars + pick + avgTrunk)` vector into `hidden_dim` space.

**Player split via `player_dim=7`:**
```python
is_black = seq_x[:, player_dim] < 0.5  # scalar[7] = isWhite
```
When both `_B.npz` (all `isWhite=0`) and `_W.npz` (all `isWhite=1`) are concatenated, this split works perfectly.

### 4.2 Cross-attention with `BidirectionalCrossMAB`

Requires both Black and White move sequences to be present in the same sample. Current `per_player=True` mode defeats this — it only sees one player per sample. Must switch to **game-level** loading (concatenate both players).

### 4.3 `DualRatingHead` — ready to use

```python
class DualRatingHead(nn.Module):
    def forward(self, z) -> Tuple[Tensor, Tensor]:  # black_rating, white_rating
```

Train targets: `meanLogPrior_B` → `black_rating`, `meanLogPrior_W` → `white_rating`.

---

## 5. `meanLogPrior` as Training Signal

`meanLogPrior = mean(log(policy[actualMove]))` measured by KataGo.

- Range: **-5.9** (20k beginner) → **-0.5** (9d professional)
- Calibrated to KataGo's perception of move quality
- Stronger players → higher (less negative) `meanLogPrior`
- Self-contained in KAB2 header — no external Elo database needed

**Bradley-Terry consistency loss:**
- `P(B stronger than W) = sigmoid(black_rating - white_rating)`
- Ground-truth "stronger" = whichever player has higher `meanLogPrior`
- Replaces the CSV Elo-based `score=1 if black wins` with an internal strength comparison

**Direct regression loss (primary):**
- `MSE(black_rating, (meanLogPrior_B - μ) / σ)` where μ/σ normalize to N(0,1)
- `σ` ≈ 1.4 across the -5.9 to -0.5 range; μ ≈ -3.2

---

## 6. Migration Plan

### Step 1: Add KAB2 loader to `data/preprocess.py`

Add `read_kab2(path)` function (see §2.3). This is the only change to existing data utilities.

### Step 2: Create `data/katago_native/dataset_kab2.py`

New dataset class `KAB2Dataset`:
- Scans a directory for `*_B.npz` + `*_W.npz` file pairs
- Reads both files, extracts `summary[2]` (meanLogPrior) as labels
- Concatenates B+W move arrays → `(n_B + n_W, moveDim)` with `xlens=[n_B+n_W]`
- Reads `trunkDim` from header dynamically → sets `input_dim` for model
- Returns `{'features': tensor, 'logprior_b': float, 'logprior_w': float, 'seq_len': int}`

Collate function: pack variable-length sequences, stack scalar labels.

### Step 3: Update `train/train_kata_native.py`

Changes:
1. Import `KAB2Dataset` instead of `KataNativeDataset`
2. Remove `feature_dim = 256`; instead read from first sample: `feature_dim = train_ds.input_dim`
3. Change model config: `enable_score=False`, `enable_dual_rating=True`
4. Pass `player_dim=7` to model forward
5. Training loss: `RatingMSELoss(black_rating, logprior_b_norm)` + `RatingMSELoss(white_rating, logprior_w_norm)` + optional Bradley-Terry consistency

### Step 4: Update `train/config_kata_native.yaml`

```yaml
model:
  input_dim: auto          # read from first KAB2 file
  hidden_dim: 128
  enable_score: false
  enable_dual_rating: true

training:
  feature_dir: "data/selfplay/kab2"   # directory with *_B.npz and *_W.npz
  label_source: "summary_meanlogprior" # not from CSV
  player_dim: 7
```

---

## 7. Obsolete Files — Cleanup

These files are superseded and should be deleted to avoid confusion:

### Delete immediately (completely superseded)

| File | Reason |
|---|---|
| `model/trunk_pick_head.py` | Old single-player model, `trunk_dim=256` `head_dim=12` hardcoded. Superseded by `dual_view.py` + `multi_task.py`. Not imported by `model/__init__.py`. |
| `train/train_trunk.py` | Ad-hoc script using old `KABN` magic, hardcoded absolute paths. Superseded by `train_kata_native.py`. |
| `train/train_trunk_final.py` | Another ad-hoc `KABN` script with hardcoded paths. Same issue. |
| `data/katago_native/extract_features.py` | Extracts 12-dim shallow features via Katago analysis subprocess. Superseded by `batch_analysis` command. |
| `data/katago_native/extract_analysis_features.py` | Persistent-process 12-dim extractor with hardcoded paths. Same. |

### Keep (still needed or reference value)

| File | Reason |
|---|---|
| `model/__init__.py`, `set_transformer.py`, `dual_view.py`, `multi_task.py`, `losses.py` | The target architecture — no changes needed |
| `data/preprocess.py` | Utilities for CSV parsing, `.npy`/`.zip` loader (may still be needed for old data migration), `normalize_features` |
| `data/katago_native/dataset.py` | Reference for collate pattern; will be partially reused in new KAB2 dataset |
| `train/train_kata_native.py` | To be updated in-place |
| `train/config_kata_native.yaml` | To be updated in-place |
| `python/` | Legacy go-strength-model scripts; separate codebase, not interfering |

---

## 8. Input Dimension Reference

For KataGo b18 network (trunkCh=512):

```
moveDim = 10 + 512 + 512 = 1034
```

For KataGo b28/b40 networks (trunkCh=384 — if used):

```
moveDim = 10 + 384 + 384 = 778
```

The `trunkDim` field in NPZHeader gives the correct value at runtime; no code should hardcode these numbers.

**Projection head:**
```python
# In DualViewSetTransformer.__init__:
self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
# input_dim = 1034 (from KAB2 header), hidden_dim = 128 (config)
```

This is the "projection head" that handles cross-model compatibility as described in `STRENGTH_MODEL_RESEARCH.md §13.2.1`.

---

## 9. Summary of Dimension Changes

| Quantity | Old (legacy) | New (KAB2) |
|---|---|---|
| Input dim | 256 (pick only) | 1034 = 10+512+512 (scalars+pick+avgTrunk) |
| Scalar dim | 12 (go-analyzer) | 10 (batch_analysis scalars) |
| Player split | interleaved B,W,B,W | `scalar[7]=isWhite`, `player_dim=7` |
| Label source | CSV BlackRating/WhiteRating (Elo) | `summary[2]` = `meanLogPrior` in KAB2 header |
| File format | `.npy` / `.zip` / `KABN` | `KAB2` (`*_B.npz`, `*_W.npz`) |
| Training unit | one player per sample | one game (both players) per sample |
| Output head | single `score_head` | `DualRatingHead` (B+W pair) |
