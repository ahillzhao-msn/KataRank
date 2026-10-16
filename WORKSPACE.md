# KataRank — Quick Start

KataRank is the analysis backend for **GoPredict**. It wraps the custom KataGo
fork (`katago-fork`) and exposes a REST API that GoPredict calls. KataGo runs
on Windows using OpenCL for best GPU performance; GoPredict itself runs in WSL/Docker.

---

## Prerequisites

- **Python 3.11+** — https://python.org/downloads
- **uv** — `pip install uv`
- **katago-fork binary** — build from [ahillzhao-msn/KataGo](https://github.com/ahillzhao-msn/KataGo)
  or download a pre-built release. Place `katago.exe` in `src\katarank\bin\` or set `KATAGO_BIN`.
- **KataGo model weights** — place a `.bin.gz` in `~/.katago/models/` for auto-discovery,
  or download from https://github.com/lightvector/KataGo/releases

---

## Install

```powershell
git clone https://github.com/ahillzhao-msn/katarank.git
cd katarank
uv sync --extra api
```

---

## Configure

On first run a config template is written to `~/.katarank/server.toml`. Edit it once:

```toml
[katarank]
# model auto-discovered from ~/.katago/models/ — or set explicitly:
# model = "C:/path/to/kata1-b18c384nbt.bin.gz"

# katago_bin = "C:/path/to/katago.exe"   # the custom fork (batch_analysis); see below
host = "127.0.0.1"
port = 8765
engine_mode = "lite"   # must match the mode clients request (GOPredict: its active model's mode)
```

`katarank-server --init-config` rewrites the commented template listing every key.

KataGo binary: `katago_bin` in `server.toml` (or `--katago-bin`) wins. Otherwise it is
auto-discovered (`katago_setup.discover_katago`) in this order, skipping stock builds that
lack `batch_analysis`:
1. `$KATAGO_BIN`
2. `src/katarank/bin/katago.exe` (bundled)
3. known install dirs (e.g. `~/katago-fork/cpp/`)
4. `PATH`

---

## Run the server

```powershell
.\scripts\start-katarank.ps1            # headless; settings from ~/.katarank/server.toml
.\scripts\start-katarank.ps1 -Restart   # after deploying new code or a new binary (refuses during training)
Invoke-RestMethod -Method Post http://127.0.0.1:8765/models/reload   # new checkpoint: no restart needed
# or in the foreground:
uv run katarank-server
```

Check it's up:

```powershell
curl http://localhost:8765/health
```

GoPredict's containers (WSL Docker) reach it through `KATARANK_HOST_URL` in GoPredict's
`.env` (default `http://winhost:8765`, where `winhost` is the WSL gateway). GoPredict's
`scripts/refresh-katarank-env.sh` keeps that `.env` entry pointed at the current gateway.

---

## Run the CLI

```powershell
# Rank a single game
uv run katarank-infer game.sgf

# See all options
uv run katarank-infer --help
uv run katarank-server --help
```

---

## Troubleshooting

**`katago binary not found`**
→ Set `katago_bin` in `~/.katarank/server.toml`, place `katago.exe` in `src\katarank\bin\`, or build from
  [ahillzhao-msn/KataGo](https://github.com/ahillzhao-msn/KataGo).

**`No KataGo model found`**
→ Place a `.bin.gz` model in `~/.katago/models/` or set `model` in `~/.katarank/server.toml`.

**Slow first start (1–5 min)**
→ Normal — KataGo loads the model into GPU memory once at boot. Subsequent requests are fast.

**`No module named 'fastapi'`**
→ Run `uv sync --extra api`.
