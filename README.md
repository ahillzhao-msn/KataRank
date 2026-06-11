# KataRank

Go player rank assessment via KataGo neural analysis.

KataRank is a Python wrapper around a customized `katago batch_analysis`
command, feeding a rank-assessment model. It uses KataGo's internal neural
representations — trunk embeddings, policy priors, and value estimates — to
predict a player's Go rank from game records, with a complete pipeline from
raw SGF files to rank prediction.

**Design principles**

1. The core is the analysis model (`KataRankModel`); SAE-based
   interpretability of its attention dimensions is on the roadmap.
2. KataGo access must be stable and efficient, stream-first.
3. Input: SGF in three forms (directory / file list / content strings),
   routed to either training or inference.
4. Inference output is the canonical `KAB2Output` (29-dim rank distribution
   per player + rating + rank); training output is the `TrainingReport`.
5. The REST API is a thin shell over the same workflow functions as the CLI.

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

## Quick start

### Install with uv

```bash
git clone <repo> katarank
cd katarank
uv sync           # creates .venv, installs katarank + dependencies
```

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

### REST API

```bash
uv run katarank-server --model kata1.bin.gz --checkpoint nets/katarank/best.pt \
    --host 0.0.0.0 --port 8765 --sgf-root /data/sgf --max-concurrency 1
```

Endpoints: `POST /rank/string | /rank/file | /rank/batch | /rank/directory`,
`GET /health`. Returns `KAB2Output` JSON. `--sgf-root` restricts file-path
endpoints to a directory; `--max-concurrency` serializes katago runs.

### Persistent engine (daemon mode)

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

### Python API

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

## Project layout

```
katarank/
├── src/katarank/
│   ├── __init__.py          public API surface
│   ├── engine.py            KataGoEngine — subprocess + stream protocol
│   ├── workflow.py          Training/Inference workflows, rank→KAB2Output
│   ├── schema.py            KAB2Sample/Batch/Output, TrainingReport, serialization
│   ├── cli.py               katarank-infer entry point
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
│   │   └── set_transformer.py  ISAB/PMA primitives (block-diagonal batch masks)
│   └── train/
│       ├── training.py      katarank-train entry point + TrainingReport emission
│       └── config_kata_native.yaml
├── tests/
├── scripts/benchmark.py
└── docs/
```

The customized KataGo lives in a separate working tree (`katago-fork/`),
modified in `cpp/command/batch_analysis.cpp`.

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

Python code in `src/katarank/`: MIT. KataGo C++ code: AGPLv3 (see
`LICENSE-GPLv3`).
