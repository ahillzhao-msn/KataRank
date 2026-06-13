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

host = "127.0.0.1"
port = 8765
engine_mode = "lite"
```

KataGo binary is auto-discovered from:
1. `src/katarank/bin/katago.exe` (bundled)
2. `~/katago-fork/cpp/katago.exe` (local build)
3. `KATAGO_BIN` env var or PATH

---

## Run the server

```powershell
uv run katarank-server
```

Check it's up:

```powershell
curl http://localhost:8765/health
```

GoPredict expects the server at `http://localhost:8765` by default
(set `KATARANK_HOST_URL` in GoPredict's `.env` if you use a different port).

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
→ Place `katago.exe` in `src\katarank\bin\`, set `KATAGO_BIN` in `.env`, or build from
  [ahillzhao-msn/KataGo](https://github.com/ahillzhao-msn/KataGo).

**`No KataGo model found`**
→ Place a `.bin.gz` model in `~/.katago/models/` or set `model` in `~/.katarank/server.toml`.

**Slow first start (1–5 min)**
→ Normal — KataGo loads the model into GPU memory once at boot. Subsequent requests are fast.

**`No module named 'fastapi'`**
→ Run `uv sync --extra api`.
