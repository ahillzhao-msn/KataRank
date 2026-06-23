# KataRank

Go player rank assessment via KataGo neural analysis.

KataRank is a Python wrapper around a customized `katago batch_analysis`
command, feeding a rank-assessment model. It uses KataGo's internal neural
representations — trunk embeddings, policy priors, and value estimates — to
predict a player's Go rank from game records, with a complete pipeline from
raw SGF files to rank prediction.

**Design principles**

1. The core is the analysis model (`KataRankModel`); SAE-based
   interpretability of its attention dimensions is scaffolded
   (`model/sae.py`, see `docs/SAE_DESIGN.md`) — SAE training lands
   with the future ReviewWorkflow.
2. KataGo access must be stable and efficient, stream-first.
3. Input: SGF in three forms (directory / file list / content strings),
   routed to either training or inference.
4. Inference output is the canonical `KAB2Output` (29-dim rank distribution
   per player + rating + rank); training output is the `TrainingReport`.
5. The REST API is a thin shell over the same workflow functions as the CLI.

## Pre-trained models

Two baseline checkpoints are included in the repository:

| Model | Checkpoint | Input | Params | rank_acc_pm1 | rating_corr | Use case |
|-------|-----------|-------|--------|-------------|-------------|----------|
| **Full** | `nets/katarank/best.pt` | 1034-dim | 2.26M | 96.4% | 0.9949 | Best accuracy (requires KataGo trunk vectors) |
| **Lite** | `nets/katarank_lite/best_lite.pt` | 10-dim | 540K | 94.8% | 0.9918 | Fast inference (scalar features only, no trunk) |

Both models predict Go player rank (20k–9d, 29 classes) from game records. The Lite model was distilled from the Full model and retains 98% of its ±1 rank accuracy while requiring only 10 scalar features per move.

**Using the models:**

```python
from katarank.model import KataRankModel

# Load either model
model = KataRankModel.load('nets/katarank/best.pt')          # Full
model = KataRankModel.load('nets/katarank_lite/best_lite.pt') # Lite

# Inference via CLI
uv run katarank-infer game.sgf --checkpoint nets/katarank/best.pt

# Inference via REST API (configured in ~/.katarank/server.toml)
curl -X POST http://localhost:8765/rank/string \
  -H "Content-Type: application/json" \
  -d '{"sgf": "(;GM[1]SZ[19];B[pd];W[dp]...)"}'
```

Training details and experiment history: see `docs/training_log.md`.

---

## Installation

### Prerequisites

- **Python** 3.10+ (3.11+ recommended)
- **PyTorch** 2.0+ (installed automatically — CUDA version auto-detected)
- **KataGo binary**: a customized `katago` build from
  [ahillzhao-msn/KataGo](https://github.com/ahillzhao-msn/KataGo/releases).
  Place the executable in `src/katarank/bin/` or pass `--katago-bin` at runtime
  (auto-detected if added to PATH).
- **KataGo model weights**: e.g., `kata1-b18c384nbt.bin.gz` from
  [KataGo releases](https://github.com/lightvector/KataGo/releases).
- **HumanSL model weights**: Required for training data generation, optional
  for inference.

### Method 1: uv (recommended)

[uv](https://docs.astral.sh/uv/) is the fastest Python package manager.

```bash
# 1. Clone the repository
git clone https://github.com/ahillzhao-msn/katarank.git
cd katarank

# 2a. Core only (CLI inference + training)
uv sync

# 2b. With API server (REST API + daemon mode)
uv sync --extra api

# 3. Verify installation
uv run katarank-infer --help
uv run katarank-server --help   # only if --extra api was used
```

### Method 2: pip + requirements.txt

```bash
git clone https://github.com/ahillzhao-msn/katarank.git
cd katarank

# Install all dependencies (core + API server)
pip install -r requirements.txt

# Or selectively:
#   Core only:      pip install numpy pyyaml torch
#   With API:       pip install 'katarank[api]'

# Verify
python -m katarank.cli --help
python -m katarank.api.server --help
```

### Method 3: Install from wheel (for deployment)

Download the `.whl` from the [releases page](https://github.com/ahillzhao-msn/katarank/releases).

```bash
pip install katarank-*.whl
# Or with API extras:
pip install 'katarank[api] @ katarank-*.whl'
```

---

## Configuration

KataRank uses a layered configuration system that auto-discovers the KataGo
binary, generates VRAM-tuned analysis configs, and accepts overrides at every
level.

### 1. KataGo binary discovery

The discovery order (first match wins):

| Priority | Source | How to set |
|----------|--------|-----------|
| 1 | `--katago-bin` CLI arg | Passed explicitly to `katarank-infer` / `katarank-server` |
| 2 | `$KATAGO_BIN` env var | `set KATAGO_BIN=C:\path\to\katago.exe` (Windows) |
| 3 | Bundled `src/katarank/bin/` | Place `katago.exe` in that directory |
| 4 | Known install directories | `~/katago-fork/release/`, `~/katago/`, etc. |
| 5 | `PATH` | `where katago` must find it |

```bash
# See exactly what the resolver finds
uv run python -c "
from katarank.katago_setup import discover_katago
try:
    print('katago:', discover_katago())
except FileNotFoundError as e:
    print(e)
"

# Test with explicit path
uv run python -c "
from katarank.katago_setup import discover_katago
print(discover_katago(explicit=r'C:\katago\katago.exe'))
"
```

The discovery also verifies the binary is a **custom fork** (stock KataGo
rejects `batch_analysis`). If it finds a stock build, it skips it and
continues searching.

### 2. VRAM-tuned analysis config (`katago analysis -config`)

The analysis daemon (**not** `batch_analysis`) requires a `.cfg` file. If
none is provided, KataRank **auto-generates one** at
`~/.katarank/analysis.cfg` based on your GPU's VRAM:

| VRAM | `numSearchThreads` | `numAnalysisThreads` | `nnMaxBatchSize` |
|------|-------------------|---------------------|-----------------|
| ≥ 16 GB | 10 | 4 | 64 |
| ≥ 8 GB | 8 | 2 | 32 |
| ≥ 4 GB | 6 | 2 | 16 |
| < 4 GB / unknown | min(CPU cores, 4) | 1 | 8 |

The auto-generated config is cached — **delete `~/.katarank/analysis.cfg`**
to regenerate after a GPU upgrade.

Override with `--config` (CLI) or `$KATAGO_CONFIG`:

```bash
uv run katarank-server --model kata1.bin.gz --config my-tuned.cfg
```

### 3. Model paths

| Model | Required | Purpose | Where to get it |
|-------|----------|---------|----------------|
| `--model` (KataGo) | **Yes** | Neural net for batch analysis | [KataGo releases](https://github.com/lightvector/KataGo/releases) |
| `--checkpoint` (KataRank) | Optional | Trained rank model `.pt` | Produced by `katarank-train` |
| `--human-model` | Training: **yes** / Inference: optional | HumanSL rank anchors for training | Custom fork's release artifacts |

### 4. Environment variables reference

| Variable | Used by | Default |
|----------|---------|---------|
| `KATAGO_BIN` | Binary discovery | auto-detected |
| `KATAGO_CONFIG` | `ensure_analysis_config()` | auto-generated `~/.katarank/analysis.cfg` |
| `CUDA_VISIBLE_DEVICES` | PyTorch / KataGo | all GPUs |
| `KATAGO_GLOBAL_ARGS` | Extra flags passed to every katago invocation | (none) |

### 5. Quick verification

```bash
# Full equipment check — binary, config, and a 1-visit end-to-end smoke test
uv run python -c "
from katarank.katago_setup import discover_katago, ensure_analysis_config, smoke_test_analysis

bin = discover_katago()
cfg = ensure_analysis_config()
model = 'path/to/kata1.bin.gz'  # substitute your path

ok, msg = smoke_test_analysis(bin, model, cfg)
print(f'Equipment check: {\"PASS\" if ok else \"FAIL\"} — {msg}')
"
```

---

## Quick start

### Generate KAB2 features from SGF files

```bash
# Full mode — trunk + scalar features (training; -human-model is REQUIRED
# for training data so rank anchors are available)
katago.exe batch_analysis \
    -model kata1-b18c384nbt.bin.gz \
    -human-model human_model.bin.gz \
    -list games.csv \
    -output-dir data/kab2/

# Stream mode — pipe KAB2 directly to Python, no disk files
katago.exe batch_analysis -model kata1.bin.gz -list games.csv -stream

# Lite mode — scalars only (10 dims), fast inference
katago.exe batch_analysis -model kata1.bin.gz -list games.csv -stream -no-trunk
```

`games.csv` columns: `File,Player Black,Player White,Score,BlackRating,WhiteRating,Set`
(`Set`: `T` = train, `V` = val, `E` = test).

### Inference (CLI)

```bash
# Engine statistics only (no checkpoint)
uv run katarank-infer game.sgf --model kata1.bin.gz

# Full model inference, archive to a single compressed file
uv run katarank-infer --sgf-dir games/ \
    --model kata1.bin.gz --checkpoint nets/katarank/best.pt \
    --output results.jsonl.gz

# From stdin
cat game.sgf | uv run katarank-infer --stdin --model kata1.bin.gz
```

Outputs `KAB2Output` JSON (one object per game): per-player 29-dim rank
distribution, rating, rank, confidence. `-human-model` is optional for
inference.

### Train the rank model

```bash
uv run katarank-train --data-dir data/kab2 --epochs 100
# or with a custom config:
uv run katarank-train --config src/katarank/train/config_kata_native.yaml
```

Training **requires** data generated with `-human-model` (HumanSL rank
anchors); it aborts otherwise. Outputs: `best.pt`, `final.pt`, and
`training_report.json` (the `TrainingReport` schema: loss curves, rank
MAE/accuracy, rating correlation, learned ordinal thresholds).

---

## REST API server

### Quick start

```bash
# Start with uv (auto-installs fastapi/uvicorn via --extra api)
uv sync --extra api
uv run katarank-server --model kata1-b18c384nbt.bin.gz \
    --checkpoint nets/katarank/best.pt \
    --host 0.0.0.0 --port 8765 --sgf-root /data/sgf --max-concurrency 1
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/rank/string` | Rank from SGF content string |
| `POST` | `/rank/file` | Rank from SGF file path |
| `POST` | `/rank/batch` | Rank multiple SGFs (file paths or strings) |
| `POST` | `/rank/directory` | Rank all `.sgf` files in a directory |
| `POST` | `/review/string` | Rank + per-move records from SGF string |
| `POST` | `/review/file` | Rank + per-move records from SGF file |
| `POST` | `/review/batch` | Rank + per-move records for multiple SGFs |
| `POST` | `/ownership/string` | Per-position ownership for each move |
| `POST` | `/variation/string` | What-if branch analysis with HumanSL profiles |
| `GET` | `/health` | Liveness check |
| `POST` | `/engine/reset` | Soft reset (clear daemon caches) |
| `POST` | `/engine/restart` | Hard restart (reload models) |

Review endpoints return `ReviewOutput` = `KAB2Output` + `moves` list
(see `docs/REVIEW_API_DESIGN.md`); request schemas are identical to `/rank/*`.

### Production deployment

#### Option A: Direct background process

```bash
# With logging to file
nohup uv run katarank-server \
    --model /path/to/kata1.bin.gz \
    --checkpoint /path/to/best.pt \
    --host 0.0.0.0 --port 8765 \
    --sgf-root /data/sgf \
    </dev/null >logs/katarank.log 2>&1 &
```

#### Option B: Windows service (NSSM)

1. Download [NSSM](https://nssm.cc/download) and place `nssm.exe` in PATH.

2. Install:

```powershell
# PowerShell (as Administrator)
.\scripts\install-service.ps1 `
    -Model "C:\models\kata1-b18c384nbt.bin.gz" `
    -Checkpoint "C:\models\katarank\best.pt" `
    -Port 8765 `
    -SgfRoot "C:\data\sgf"
```

3. Manage:

```cmd
net start KataRank       :: start
net stop KataRank        :: stop
nssm status KataRank     :: check status
nssm remove KataRank confirm  :: uninstall
```

#### Option C: Use katarank-server.bat

```cmd
scripts\katarank-server.bat --model kata1.bin.gz --host 0.0.0.0 --port 8765
```

---

## Persistent engine (daemon mode)

`katago batch_analysis -daemon` keeps models loaded and accepts jobs via
stdin (one line = path to a games.csv; `reset` clears NN caches; `quit`
exits). Each job's frames end with a `0x01` marker; `0x00` means the daemon
exited. The REST server uses this by default (`--no-persistent` to opt out),
dropping request latency from ~tens of seconds (model load) to analysis time.

```python
from katarank import PersistentKataGoEngine

with PersistentKataGoEngine(model='kata1.bin.gz', mode='lite') as eng:
    list(eng.stream_games(['g1.sgf']))   # ~1-2 s, no model reload
    list(eng.stream_games(['g2.sgf']))
    eng.soft_reset()                     # clear NN caches, keep models (~ms)
    eng.restart()                        # hard reset: new process, reload models
    eng.close(force=True)                # kill a wedged daemon
```

REST control: `POST /engine/reset` (soft), `POST /engine/restart` (hard),
`GET /health` reports daemon liveness.

---

## Python API

```python
from katarank import KataGoEngine, run_rank_files
from katarank.model import KataRankModel
from katarank.workflow import InferenceWorkflow

engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')

# Stream KAB2 features directly from KataGo — no disk I/O
for side, moves, info in engine.stream_games(['game.sgf'], mode='lite'):
    print(info['game_id'], side, moves.shape, info['mean_log_prior'])

# Paired per-game tensors (B/W matched by game id)
for x_b, x_w, info_b, info_w in engine.stream_to_tensors(['game.sgf']):
    ...

# Model inference → KAB2Output
model = KataRankModel.load('nets/katarank/best.pt')
wf = InferenceWorkflow(model, engine)
outputs = run_rank_files(engine, wf, ['game1.sgf', 'game2.sgf'])

# File-based batch (writes combined <stem>.npz per game + _meta.csv)
engine.batch_to_files(output_dir='data/kab2/', sgf_paths=['game1.sgf'])
```

---

## Build from source

```bash
# Build wheel
uv build

# Output: dist/katarank-0.2.0-py3-none-any.whl

# Install the wheel
pip install dist/katarank-0.2.0-py3-none-any.whl
# With API extras:
pip install 'katarank[api] @ dist/katarank-0.2.0-py3-none-any.whl'

# Or publish to PyPI (requires API token)
# uv publish
```

---

## Output formats

### File mode — combined KAB2 (one file per game)

`<sgf-stem>.npz` = `[4B B_size][B KAB2 payload][4B W_size][W KAB2 payload]`,
each payload zlib-compressed. A `_meta.csv` with per-game summaries is
written alongside.

KAB2 payload layout:

```
Offset  Size  Field
     0     4  magic b'KAB2'
     4     4  numMoves         (int32)
     8     4  scalarDim        (int32, = 10)
    12     4  trunkDim         (int32, 0 in lite mode)
    16     4  pickDim          (int32, = trunkDim)
    20     4  nnXLen           (int32)
    24     4  nnYLen           (int32)
    28     4  flags            (int32, bit0 = zlib compressed)
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

### Stream mode — per-player frames with game id

```
[1 byte side 'B'/'W'][4B uint32 idLen][game id][4B uint32 size][KAB2 payload]
```

One frame per player per game (uncompressed), terminated by a single `0x00`
byte. The game id is the SGF filename stem, letting the Python reader pair
B/W frames and map results back to inputs. Progress goes to stderr, binary
data to stdout.

---

## Architecture overview

```
SGF files / directory / strings
   │
   ▼
katago batch_analysis           ← customized C++ binary (katago-fork)
   │  per-move features, two delivery modes:
   │    file:   combined <stem>.npz per game (compressed, archival)
   │    stream: per-player frames with game id (uncompressed, pipe)
   ▼
KAB2Dataset / KAB2StreamDataset ← data pipeline (BaseKAB2Dataset contract)
   │
   ▼
KataRankModel                   ← DualViewSetTransformer + ordinal heads
   │  causal B/W cross-attention; 29 rank classes (20k–9d)
   ▼
KAB2Output (inference)  /  TrainingReport (training)
```

---

## Project layout

```
katarank/
├── src/katarank/
│   ├── __init__.py          public API surface
│   ├── engine.py            KataGoEngine — subprocess + stream protocol
│   ├── workflow.py          Training/Inference workflows, rank→KAB2Output
│   ├── schema.py            KAB2Sample/Batch/Output, TrainingReport, serialization
│   ├── cli.py               katarank-infer entry point
│   ├── katago_setup.py      KataGo binary discovery + VRAM auto-config
│   ├── analysis_daemon.py   Persistent `katago analysis` (JSON protocol)
│   ├── api/server.py        katarank-server (FastAPI, thin shell over workflow)
│   ├── data/
│   │   ├── preprocess.py    read_kab2(), read_kab2_combined(), probe_kab2_dim()
│   │   └── datasets/
│   │       ├── dataset_kab2.py    KAB2Dataset (combined .npz files)
│   │       └── dataset_stream.py  KAB2StreamDataset (live engine stream)
│   ├── model/
│   │   ├── dual_view.py     DualViewSetTransformer (causal cross-attention)
│   │   ├── multi_task.py    KataRankModel, OrdinalLogisticHead
│   │   ├── losses.py        KataRankLoss (rating MSE + Bradley-Terry + rank anchor)
│   │   ├── set_transformer.py  ISAB/PMA primitives (block-diagonal batch masks)
│   │   ├── interpret.py     ActivationCapture — forward-hook activation capture
│   │   └── sae.py           SAE interface: FeatureExtractor, FeatureRegistry
│   └── train/
│       ├── training.py      katarank-train entry point + TrainingReport emission
│       └── config_kata_native.yaml
├── scripts/
│   ├── katarank-server.bat  Windows startup script
│   ├── install-service.ps1  Windows NSSM service installer
│   └── benchmark.py         Performance benchmark
├── tests/
├── docs/
├── requirements.txt         Pinned runtime deps (core + API server)
├── pyproject.toml           Project metadata + build config
└── README.md
```

The customized KataGo subcommands live in a separate project
(`ahillzhao-msn/KataGo`), with `batch_analysis` additions in
`cpp/command/batch_analysis.cpp`.

---

## Model details

### DualViewSetTransformer

- Black and White move streams encoded separately (ISAB set encoders)
- Causal cross-attention: each player attends only to opponent moves already
  played (turn parity derived from per-stream position)
- Block-diagonal attention masks keep games in a packed batch independent
- `StreamPooling`: opening/midgame/endgame segmented attention pooling

### KataRankLoss

| Component | Description |
|-----------|-------------|
| `RatingMSELoss` | MSE between predicted rating and `meanLogPrior` |
| `BradleyTerry` | Pairwise B-vs-W ordering consistency |
| `RankAnchorLoss` | Ordinal NLL vs `humanRankIdx`, confidence-weighted by `humanLogPrior`, auto-zero when label = –1 |

Convention: `loss_fn(predictions, targets) -> dict` with key `'total'`.

---

## Requirements

- Python 3.10+, PyTorch 2.0+
- A compiled customized `katago` binary (katago-fork)
- KataGo model weights (e.g. `kata1-b18c384nbt.bin.gz`)
- HumanSL model weights — **required for training data generation**, optional
  for inference

## License

MIT — see `LICENSE`.
