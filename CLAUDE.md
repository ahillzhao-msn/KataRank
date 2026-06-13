# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (core + API server)
uv sync --extra api

# Run tests
uv run pytest tests/ -v --tb=short

# Run a single test class or method
uv run pytest tests/test_model.py::TestKAB2Parsing -v
uv run pytest tests/test_model.py::TestKataRankModel::test_save_load_roundtrip -v

# CLI entry points
uv run katarank-infer --help
uv run katarank-train --help
uv run katarank-server --help

# Build wheel
uv build
```

## Architecture

KataRank wraps a **custom-fork KataGo binary** (`batch_analysis` subcommand, not in stock KataGo) and feeds its output into a PyTorch rank-prediction model. The pipeline is:

```
SGF files → katago batch_analysis (C++ fork) → KAB2 binary frames
    → KataGoEngine (Python stream reader) → KataRankModel → KAB2Output
```

### KAB2 binary format

The custom KataGo binary emits per-player feature frames in two delivery modes:

- **Stream mode** (pipe): `[1B side B/W][4B idLen][game_id][4B size][KAB2 payload]`, terminated by `0x00`. Used for live inference.
- **File mode** (disk): `[4B B_size][B payload][4B W_size][W payload]` packed into `.npz` files. Used for training datasets.

Each KAB2 payload is a binary struct with a 96-byte header (magic `KAB2`, dims, 16×float32 `PlayerSummary`) followed by `numMoves × moveDim` float32 move features. Key `PlayerSummary` indices: `[2]` = `meanLogPrior` (primary training target), `[10]` = `humanRankIdx` (0–28 for 20k–9d, –1 if absent), `[11]` = `humanLogPrior`. Parsing lives in `engine.py:parse_kab2_buffer`.

### Core modules

| Module | Role |
|--------|------|
| `engine.py` | `KataGoEngine` — launches and reads from the katago subprocess; `PersistentKataGoEngine` keeps the daemon alive between requests |
| `workflow.py` | `InferenceWorkflow` / `TrainingWorkflow`; `run_rank_files`, `run_rank_strings`, `run_review_files`; `_move_records` for per-move review output |
| `schema.py` | `KAB2Sample`, `KAB2Output`, `TrainingReport`; collate/serialize helpers |
| `katago_setup.py` | Binary discovery (5-tier priority) and VRAM-based auto-generation of `~/.katarank/analysis.cfg` |
| `analysis_daemon.py` | Persistent `katago analysis` JSON-protocol daemon used by the REST server |
| `api/server.py` | FastAPI thin shell over the same workflow functions as the CLI |

### Model (`src/katarank/model/`)

- **`DualViewSetTransformer`** (`dual_view.py`): encodes Black and White move streams independently via ISAB set encoders, then applies causal cross-attention (each player attends only to already-played opponent moves). Block-diagonal attention masks keep games independent when batch-packed.
- **`KataRankModel`** (`multi_task.py`): wraps `DualViewSetTransformer` + `OrdinalLogisticHead`; outputs `b_rating`, `w_rating`, `rank_probs_b`, `rank_probs_w` (29-class softmax, 20k–9d).
- **`KataRankLoss`** (`losses.py`): three components — `RatingMSELoss` (vs `meanLogPrior`), `BradleyTerry` (B-vs-W pairwise ordering), `RankAnchorLoss` (ordinal NLL vs `humanRankIdx`, confidence-weighted by `humanLogPrior`, auto-zeroed when label = –1). Convention: `loss_fn(predictions, targets) -> dict` with key `'total'`.
- **`ActivationCapture`** (`interpret.py`): forward-hook context manager for capturing residual stream activations and attention maps; supports `accumulate=True` for corpus collection.
- **`SparseAutoencoder`** / **`FeatureExtractor`** / **`FeatureRegistry`** (`sae.py`): SAE interpretability scaffolding over the cross-attention sites (see `docs/SAE_DESIGN.md`).

### Data pipeline (`src/katarank/data/`)

- **`KAB2Dataset`**: reads combined `.npz` files (file mode) for offline training.
- **`KAB2StreamDataset`**: wraps a live `KataGoEngine` stream for online training.
- Both implement `BaseKAB2Dataset`. Preprocessing helpers: `read_kab2`, `read_kab2_combined`, `probe_kab2_dim` in `data/preprocess.py`.

### Batching convention

Games are packed into a single tensor: `x` is `(total_moves_across_games, input_dim)`, `xlens` is a list of per-game move counts. `pack_batch` / `unbatch` utilities in `model/__init__.py`. The `isWhite` flag lives at column 7 of each move row.

### Key external dependency

Requires the **custom KataGo fork** binary (`ahillzhao-msn/KataGo`). Stock KataGo does not have `batch_analysis`. Binary is gitignored — place at `src/katarank/bin/katago.exe` or set `KATAGO_BIN` env var. The `katago analysis` daemon (used by REST server) also requires this fork.

### Tests

All tests are in `tests/test_model.py`. Tests are pure-Python/PyTorch and do **not** require a real KataGo binary — the engine is stubbed with synthesized KAB2 bytes or `_StubEngine`. The `TestBatchIndependence` class is a regression suite for the critical batch-isolation invariant (games must not leak attention across batch boundaries).
