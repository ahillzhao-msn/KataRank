# KataRank Workspace Setup

Quick guide to get a working environment from scratch.

---

## Recommended directory layout

```
workspace/
├── katarank/                    ← this repository
│   ├── src/katarank/bin/        ← katago.exe here (or use KATAGO_BIN)
│   ├── .env                     ← your local config (gitignored)
│   ├── logs/                    ← service logs (created on first run)
│   └── ...
├── models/
│   ├── kata1-b18c384nbt.bin.gz  ← KataGo b18 model
│   └── human_model.bin.gz       ← HumanSL model (optional)
├── checkpoints/
│   └── katarank/best.pt         ← trained rank model (optional)
└── sgf/                         ← game records
    └── ...
```

---

## Step 1 — Clone and install

```powershell
git clone https://github.com/ahillzhao-msn/katarank.git
cd katarank

# Core + API server
uv sync --extra api

# Verify
uv run katarank-infer --help
```

If you don't have `uv`: `pip install uv` or see https://docs.astral.sh/uv/

---

## Step 2 — Get KataGo binary

Download the custom fork binary from [GitHub Releases](https://github.com/ahillzhao-msn/KataGo/releases).

| Platform | File |
|----------|------|
| Windows (CUDA) | `katago.exe` |
| Windows (OpenCL) | `katago-opencl.exe` |
| Linux (OpenCL) | `katago` |
| macOS (Metal) | `katago-metal` |

Place in `src/katarank/bin/katago.exe` **or** set `KATAGO_BIN` in `.env`.

> **Stock KataGo won't work** — the fork adds `batch_analysis` which is required.

---

## Step 3 — Get model weights

### KataGo neural network
```
https://github.com/lightvector/KataGo/releases
```
Recommended: `kata1-b18c384nbt-s9986952192-d4519031827.bin.gz`

### HumanSL model (optional — needed for training data)
Provided in the KataGo fork release artifacts as `human_model.bin.gz`.

---

## Step 4 — Configure

```powershell
# Copy the template
cp .env.example .env

# Edit .env with your paths:
#   KATAGO_BIN=C:\path\to\katago.exe
#   KATAGO_MODEL=C:\models\kata1-b18c384nbt.bin.gz   (optional — pass via CLI)
```

---

## Step 5 — Verify equipment

```powershell
uv run python -c "
from katarank.katago_setup import discover_katago, ensure_analysis_config
bin_path = discover_katago()
print('KataGo binary:', bin_path)
cfg_path = ensure_analysis_config()
print('Analysis config:', cfg_path)
"
```

Expected output:
```
KataGo binary: C:\...\katago.exe
Analysis config: C:\Users\<you>\.katarank\analysis.cfg
```

If binary discovery fails, set `KATAGO_BIN` in `.env` or pass `--katago-bin`.

---

## Step 6 — Start the API server

```powershell
uv run katarank-server `
    --model C:\models\kata1-b18c384nbt.bin.gz `
    --host 0.0.0.0 `
    --port 8765

# Check health
curl http://localhost:8765/health
```

### As a Windows service (auto-start on boot)

```powershell
# Requires NSSM: https://nssm.cc/download
# Run as Administrator
.\scripts\install-service.ps1 `
    -Model "C:\models\kata1-b18c384nbt.bin.gz" `
    -Port 8765

net start KataRank
```

---

## Environment variable reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KATAGO_BIN` | No | auto-detect | Path to katago binary |
| `KATAGO_CONFIG` | No | auto-generate | Path to analysis .cfg |
| `CUDA_VISIBLE_DEVICES` | No | all GPUs | GPU selection |
| `KATAGO_GLOBAL_ARGS` | No | (none) | Extra flags for katago |
| `KATARANK_HOST_URL` | No | http://localhost:8765 | Used by gopredict frontend |

Full reference: see `.env.example`.

---

## Troubleshooting

**`FileNotFoundError: katago binary not found`**  
→ Set `KATAGO_BIN` in `.env` or place the binary in `src/katarank/bin/`.

**`RuntimeError: katago analysis subprocess exited immediately`**  
→ Run `katago.exe runtests` to verify the binary. Check if a GPU driver is available.  
→ Delete `~/.katarank/analysis.cfg` to regenerate a config tuned to your VRAM.

**`ModuleNotFoundError: No module named 'fastapi'`**  
→ Run `uv sync --extra api` (the `api` extra is required for the server).

**Slow first request (10–30 s)**  
→ Normal — KataGo loads models into GPU memory on the first query. Subsequent queries are fast because the daemon keeps models loaded.

**Port already in use**  
→ Change `--port` or stop the existing process: `net stop KataRank`.
