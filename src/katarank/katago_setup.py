"""
KataGo equipment QA — discovery, config provisioning, verification.

KataRank depends on KataGo's custom `batch_analysis` subcommand (it emits
KAB2 frames; stock KataGo does not have it). This module guarantees the
engine is correctly equipped before any daemon starts:

  1. discover_katago()      — resolve the binary (explicit → env → bundled
                              → known install dirs → PATH) and verify
                              batch_analysis capability.
  2. ensure_analysis_config() — `katago analysis` REQUIRES -config; when the
                                caller provides none, generate a VRAM-aware
                                config (cached at ~/.katarank/analysis.cfg).
  3. smoke_test_analysis()  — optional end-to-end check: one 1-visit query
                              through a real `katago analysis` process.

Ported from go-analyzer's built-in discovery.py / tuning.py (now archived);
live candidate benchmarking was dropped in favour of a static VRAM tier
table — the b28 model costs ~90 s of load time per candidate, which makes
per-boot live tuning a net loss for a long-lived server.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0

_EXE = 'katago.exe' if sys.platform == 'win32' else 'katago'


# ─── 1. Binary discovery ─────────────────────────────────────────────────────

def _candidate_binaries() -> list:
    home = Path.home()
    return [
        Path(__file__).parent / 'bin' / _EXE,          # bundled with katarank
        home / 'katago-fork' / 'release' / _EXE,       # custom build
        home / 'katago-fork' / 'cpp' / _EXE,
        home / 'katago' / _EXE,
        home / 'katago' / 'opencl' / _EXE,
    ]


def has_batch_analysis(katago_bin: str, timeout: float = 15.0) -> bool:
    """True if the binary supports the `batch_analysis` subcommand.

    Stock katago answers 'Unknown subcommand: batch_analysis'; a custom build
    recognises the subcommand (and merely rejects the probe argument), so
    the distinguishing signal is the stderr text, not the exit code.
    """
    try:
        r = subprocess.run(
            [katago_bin, 'batch_analysis', '-help'],
            capture_output=True, text=True, timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
        out = (r.stdout + r.stderr).lower()
        return 'unknown subcommand' not in out
    except (OSError, subprocess.TimeoutExpired):
        return False


def discover_katago(explicit: Optional[str] = None,
                    require_fork: bool = True) -> str:
    """Resolve the KataGo binary, verifying batch_analysis support.

    Resolution order: explicit arg → $KATAGO_BIN → bundled bin/ →
    known install dirs → PATH. Raises FileNotFoundError with an
    actionable message when nothing qualifies.
    """
    tried = []

    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get('KATAGO_BIN', '')
    if env:
        candidates.append(Path(env))
    candidates += _candidate_binaries()
    which = shutil.which('katago')
    if which:
        candidates.append(Path(which))

    for cand in candidates:
        if not cand.exists():
            tried.append(f'{cand} (missing)')
            continue
        if require_fork and not has_batch_analysis(str(cand)):
            tried.append(f'{cand} (stock katago — no batch_analysis)')
            continue
        logger.info('katago binary: %s', cand)
        return str(cand)

    raise FileNotFoundError(
        'No usable KataGo fork binary found. Tried:\n  '
        + '\n  '.join(tried) +
        '\nFix: build the fork (github.com/ahillzhao-msn/KataGo) and either '
        'place katago.exe in src/katarank/bin/, set KATAGO_BIN, or pass '
        '--katago-bin.'
    )


# ─── 2. VRAM-aware analysis config ───────────────────────────────────────────

def guess_vram_mb() -> Optional[int]:
    """Total GPU memory in MB via nvidia-smi, or None if undetectable."""
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split('\n')[0].strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def analysis_tuning(vram_mb: Optional[int]) -> dict:
    """Static tier table (from go-analyzer's tested candidates).

    Picks the mid/aggressive tier per VRAM class — safe headroom while the
    GPU is shared with the batch_analysis daemon.
    """
    cpu = os.cpu_count() or 8
    if vram_mb and vram_mb >= 16384:
        return {'numSearchThreads': 10, 'numAnalysisThreads': 4,
                'nnMaxBatchSize': 64}
    if vram_mb and vram_mb >= 8192:
        return {'numSearchThreads': 8, 'numAnalysisThreads': 2,
                'nnMaxBatchSize': 32}
    if vram_mb and vram_mb >= 4096:
        return {'numSearchThreads': 6, 'numAnalysisThreads': 2,
                'nnMaxBatchSize': 16}
    return {'numSearchThreads': min(cpu, 4), 'numAnalysisThreads': 1,
            'nnMaxBatchSize': 8}


def ensure_analysis_config(config: Optional[str] = None,
                           cache_dir: Optional[str] = None) -> str:
    """Return a valid -config path for `katago analysis`.

    If the caller provided one, validate it exists. Otherwise generate a
    VRAM-aware config once and cache it (~/.katarank/analysis.cfg).
    """
    if config:
        p = Path(config)
        if not p.exists():
            raise FileNotFoundError(f'katago config not found: {config}')
        return str(p)

    cache = Path(cache_dir) if cache_dir else Path.home() / '.katarank'
    cfg_path = cache / 'analysis.cfg'
    if cfg_path.exists():
        return str(cfg_path)

    vram = guess_vram_mb()
    tune = analysis_tuning(vram)
    cache.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        '# Auto-generated by katarank.katago_setup (vram=%s MB)\n'
        '# Delete this file to regenerate after a hardware change.\n'
        'logToStderr = true\n'
        'reportAnalysisWinratesAs = BLACK\n'
        'maxVisits = 100\n'
        'numSearchThreads = %d\n'
        'numAnalysisThreads = %d\n'
        'nnMaxBatchSize = %d\n'
        % (vram, tune['numSearchThreads'], tune['numAnalysisThreads'],
           tune['nnMaxBatchSize']),
        encoding='utf-8',
    )
    logger.info('generated analysis config: %s (vram=%s MB, %s)',
                cfg_path, vram, tune)
    return str(cfg_path)


# ─── 2b. Model discovery ────────────────────────────────────────────────────

def discover_model(explicit: Optional[str] = None) -> str:
    """Resolve the best available KataGo model (.bin.gz).

    Resolution order:
      1. explicit argument (--model CLI flag)
      2. $KATAGO_MODEL env var
      3. ~/.katago/models/  — largest .bin.gz (proxy for strongest network)
      4. ~/.katarank/models/
    Raises FileNotFoundError with an actionable message when nothing is found.
    """
    candidates: list[Optional[str]] = [
        explicit,
        os.environ.get('KATAGO_MODEL'),
    ]
    for c in candidates:
        if c and Path(c).exists():
            logger.info('katago model: %s', c)
            return c

    search_dirs = [
        Path.home() / '.katago' / 'models',
        Path.home() / '.katarank' / 'models',
    ]
    best: Optional[Path] = None
    for d in search_dirs:
        if not d.is_dir():
            continue
        models = sorted(d.glob('*.bin.gz'), key=lambda p: p.stat().st_size, reverse=True)
        if models:
            best = models[0]
            break

    if best:
        logger.info('katago model (auto-discovered): %s', best)
        return str(best)

    raise FileNotFoundError(
        'No KataGo model found. Tried: explicit arg, $KATAGO_MODEL, '
        '~/.katago/models/, ~/.katarank/models/.\n'
        'Fix: place a .bin.gz model in ~/.katago/models/ or pass --model.'
    )


# ─── 3. Smoke verification ───────────────────────────────────────────────────

def smoke_test_analysis(katago_bin: str, model: str, config: str,
                        timeout: float = 180.0) -> Tuple[bool, str]:
    """One 1-visit query through a real `katago analysis` process.

    Heavy (full model load) — intended for `katarank-verify`-style manual
    checks or CI, not for every server boot.
    """
    query = json.dumps({
        'id': 'smoke', 'moves': [['B', 'Q16']], 'rules': 'chinese',
        'komi': 7.5, 'boardXSize': 19, 'boardYSize': 19, 'maxVisits': 1,
        'analyzeTurns': [1],
    }) + '\n'
    try:
        proc = subprocess.Popen(
            [katago_bin, 'analysis', '-model', model, '-config', config],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=_CREATE_NO_WINDOW,
        )
        proc.stdin.write(query.encode())
        proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors='replace')[-500:]
                return False, f'katago exited (code {proc.returncode}): {err}'
            line = proc.stdout.readline().decode(errors='replace').strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get('id') == 'smoke':
                proc.kill()
                return ('rootInfo' in obj or 'moveInfos' in obj), 'ok'
        proc.kill()
        return False, f'no response within {timeout}s'
    except (OSError, json.JSONDecodeError) as e:
        return False, str(e)
