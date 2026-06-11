# KataRank

Go player rank assessment via KataGo neural analysis.

KataRank uses KataGo's internal neural network representations — trunk embeddings, policy priors, and value estimates — to predict a player's Go rank from game records. It runs a complete pipeline from raw SGF files to rank prediction without requiring human-provided labels for inference.

---

## Architecture overview

```
SGF files
   │
   ▼
katago batch_analysis          ← customized C++ binary (katago/ subdir)
   │  extracts per-move features → KAB2 binary format
   ▼
KAB2Dataset (PyTorch)          ← data pipeline
   │  reads _B.npz / _W.npz file pairs
   ▼
DualViewSetTransformer          ← model/dual_view.py
   │  causal cross-attention between Black and White move streams
   ▼
KataRankModel                  ← model/multi_task.py
   │  dual rating heads + ordinal rank heads (20k–9d, 29 classes)
   ▼
rank prediction
```

### KAB2 binary format

Each game produces two files: `<stem>_B.npz` (Black) and `<stem>_W.npz` (White).

```
Offset  Size  Field
     0     4  magic b'KAB2'
     4     4  numMoves         (int32)
     8     4  scalarDim        (int32, = 10)
    12     4  trunkDim         (int32, 0 in lite mode)
    28     4  flags            (bit0 = zlib compressed)
    32    64  PlayerSummary    (16 × float32)
    96     –  move records     (numMoves × moveDim float32)
```

`moveDim = scalarDim + 2 × trunkDim`

Key `PlayerSummary` fields:

| Index | Field | Description |
|-------|-------|-------------|
| 2 | `meanLogPrior` | Primary training target (log KataGo policy prior) |
| 10 | `humanRankIdx` | 0–28 (20k–9d), –1 if HumanSL not computed |
| 11 | `humanLogPrior` | HumanSL confidence weight |

---

## Quick start

### Install with uv

```bash
git clone <repo> katarank
cd katarank
uv sync           # creates .venv, installs katarank + dependencies
```

### Generate KAB2 features from SGF files

```bash
# Full mode — trunk + scalar features (for training)
katago/katago.exe batch_analysis \
    -model kata1-b18c384nbt.bin.gz \
    -list games.csv \
    -output-dir data/kab2/

# Lite mode — scalars only (fast, for quick inference)
katago/katago.exe batch_analysis \
    -model kata1-b18c384nbt.bin.gz \
    -list games.csv \
    -output-dir data/kab2/ \
    -no-trunk

# Stream mode — pipe KAB2 directly to Python, no disk files
katago.exe batch_analysis \
    -model kata1-b18c384nbt.bin.gz \
    -list games.csv \
    -stream

# Lite mode — scalars only (10 dims), 100x faster, no trunk
katago.exe batch_analysis \
    -model kata1-b18c384nbt.bin.gz \
    -list games.csv \
    -stream -no-trunk

# Stream mode protocol: [1 byte side][4 bytes size][KAB2 payload] per player,
# terminated by 0x00. Progress goes to stderr, binary data to stdout.
```

`games.csv` format: `File,Player Black,Player White,BlackRating,WhiteRating,Set`  
`Set` column: `T` = train, `V` = val, `E` = test.

### Train the rank model

```bash
uv run katarank-train \
    --data-dir data/kab2 \
    --epochs 100

# or with a custom config:
uv run katarank-train --config src/katarank/train/config_kata_native.yaml
```

### Python API

```python
from katarank import KataGoEngine
from katarank.model import KataRankModel

# Stream KAB2 features directly from KataGo — no disk I/O
engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')
for side, moves, info in engine.stream_games(['game.sgf']):
    print(side, moves.shape, info['mean_log_prior'])
    # side='B'|'W', moves=(N,10) in lite mode or (N,10+2*trunkCh) in full

# Load a trained model and run inference
rank_model = KataRankModel.load('nets/katarank/best.pt')
for x_b, x_w, info_b, info_w in engine.stream_to_tensors(['game.sgf']):
    import torch
    x = torch.cat([x_b, x_w], dim=0)
    out = rank_model(x, xlens=[len(x_b) + len(x_w)])
    # out['b_log_prior'], out['w_log_prior']  — rating signals
    # out['b_rank_logits'], out['w_rank_logits']  — 29-class rank distribution

# File-based batch (writes _B.npz / _W.npz to output_dir)
engine.batch_to_files(['game1.sgf', 'game2.sgf'], output_dir='data/kab2/')
```

---

## Project layout

```
katarank/
├── src/
│   └── katarank/
│       ├── __init__.py          KataGoEngine, parse_kab2_buffer
│       ├── engine.py            subprocess wrapper for batch_analysis
│       ├── model/
│       │   ├── dual_view.py     DualViewSetTransformer (causal cross-attention)
│       │   ├── multi_task.py    KataRankModel, OrdinalLogisticHead
│       │   ├── losses.py        KataRankLoss, BradleyTerry, RatingMSELoss
│       │   ├── set_transformer.py  Set Transformer primitives (ISAB, PMA, …)
│       │   └── archive/         v2 model files (reference only)
│       ├── data/
│       │   ├── preprocess.py    read_kab2(), probe_kab2_dim()
│       │   └── katago_native/
│       │       ├── dataset_kab2.py  KAB2Dataset, kab2_collate, make_kab2_loader
│       │       ├── selfplay.py      SelfPlayEngine for data generation
│       │       └── weight_manager.py  KataGo weight downloading
│       └── train/
│           ├── train_kata_native.py  training entry point
│           └── config_kata_native.yaml
├── katago/                      KataGo C++ source (git submodule)
│   └── cpp/command/batch_analysis.cpp  ← customized for KAB2 output
├── tests/
├── docs/
├── pyproject.toml
└── uv.lock
```

---

## Model details

### DualViewSetTransformer

Processes Black and White move sequences separately with causal cross-attention:

- Black moves attend to all White moves played *before* the current turn
- White moves attend to all Black moves played *up to and including* the current turn
- Causal masks prevent information leakage across time
- `StreamPooling` aggregates the attended sequence into a per-player embedding

### KataRankModel

```
input (N_moves, input_dim)
    ↓
DualViewSetTransformer
    ↓                    ↓
B embedding         W embedding
    ↓                    ↓
rating head (MLP)   rating head (MLP)
    ↓                    ↓
b_log_prior         w_log_prior
    ↓                    ↓
ordinal head        ordinal head     (29 classes: 20k → 9d)
```

### KataRankLoss

Combines three signals:

| Component | Weight | Description |
|-----------|--------|-------------|
| `RatingMSELoss` | 1.0 | MSE between predicted and `meanLogPrior` |
| `BradleyTerry` | 0.5 | Pairwise consistency across the batch |
| `RankAnchorLoss` | 0.3 | Ordinal cross-entropy, confidence-weighted by `humanLogPrior`, skipped when `humanRankIdx = -1` |

---

## KataGo C++ customizations

The `katago/` subdir contains a modified KataGo build. Changes relative to upstream are in `katago/cpp/command/batch_analysis.cpp`:

- **KAB2 format** — custom binary output (96-byte header + move records)
- **HumanSL second pass** — optional `humanRankIdx` / `humanLogPrior` annotation via `-human-model`
- **Stream mode** (`-stream`) — length-prefixed KAB2 frames on stdout; no disk I/O
- **Lite mode** (`-no-trunk`) — scalar-only output (10 dims vs 10 + 2×trunkCh)
- **SGF filename** — output files named after the source SGF stem
- **`_meta.csv`** — per-game metadata CSV alongside the .npz files

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- A compiled `katago` binary (see `katago/Compiling.md`)
- KataGo model weights (e.g. `kata1-b18c384nbt.bin.gz`)
- Optional: HumanSL model weights for rank annotation during feature extraction

---

## License

The Python code in `src/katarank/` is released under the MIT License.  
The KataGo C++ code in `katago/` is licensed under AGPLv3 (see `LICENSE-GPLv3`).
