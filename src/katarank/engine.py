"""
KataRank — KataGo Engine Interface
=====================================
Python wrapper around the `katago batch_analysis` binary.

Two analysis modes:
  'full'    — trunk + pick vectors, for training
  'lite'    — scalars only (10 dims), for fast inference

Two delivery modes:
  stream    — uncompressed KAB2 frames via stdout pipe (zero disk I/O),
              one frame per player tagged with the game id (SGF stem):
              [1B side][4B idLen][game id][4B size][payload], 0x00 terminator
  file      — one combined compressed <stem>.npz per game to output_dir:
              [4B B_size][B KAB2][4B W_size][W KAB2]

Input types supported:
  - SGF file paths (list or single string)
  - SGF directory (scans *.sgf)
  - SGF content strings (written to temp files)
  - CSV list file (pass-through to -list)

Typical usage:

    engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')

    # Stream from file paths (lite mode)
    for side, moves, info in engine.stream_games(['g1.sgf', 'g2.sgf'], mode='lite'):
        print(side, moves.shape, info['mean_log_prior'])

    # Stream from SGF strings
    for x_b, x_w, info_b, info_w in engine.stream_to_tensors(
        sgf_strings=[sgf_content], mode='lite'
    ):
        out = model(x_b)  # immediate inference, zero disk I/O

    # Stream from directory
    for side, moves, info in engine.stream_games(sgf_dir='./sgfs/'):
        ...

    # File output
    engine.batch_to_files(sgf_paths=['g1.sgf', 'g2.sgf'], output_dir='features/')
"""

import os
import struct
import subprocess
import sys
import tempfile
import threading
import zlib
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generator, Iterable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from katarank.analysis_daemon import AnalysisDaemon

import numpy as np

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0


# ─── KAB2 constants ──────────────────────────────────────────────────────────

_MAGIC          = b'KAB2'
_HEADER_SIZE    = 96
_SUMMARY_OFFSET = 32

# PlayerSummary field indices
_SUM_MEAN_LP      = 2
_SUM_HUMAN_RANK   = 10
_SUM_HUMAN_LP     = 11

# Stream protocol: [1B side][4B uint32 idLen][game id][4B uint32 size][payload]
_TERMINATOR = b'\x00'   # daemon/process exit
_JOB_DONE   = b'\x01'   # end of one daemon job


# ─── Low-level KAB2 parser (bytes → ndarray) ──────────────────────────────────

def parse_kab2_buffer(data: bytes) -> Tuple[np.ndarray, Dict]:
    """Parse a KAB2 payload from raw bytes (in-memory, no file I/O)."""
    if data[:4] != _MAGIC:
        raise ValueError(f"Not a KAB2 buffer (magic={data[:4]!r})")

    num_moves  = struct.unpack_from('<i', data, 4)[0]
    scalar_dim = struct.unpack_from('<i', data, 8)[0]
    trunk_dim  = struct.unpack_from('<i', data, 12)[0]
    flags      = struct.unpack_from('<i', data, 28)[0]
    summary    = struct.unpack_from('<16f', data, _SUMMARY_OFFSET)

    payload = data[_HEADER_SIZE:]
    if flags & 1:
        clen = struct.unpack_from('<I', payload, 0)[0]
        payload = zlib.decompress(payload[4: 4 + clen])

    move_dim = scalar_dim + 2 * trunk_dim
    moves = np.frombuffer(payload, dtype=np.float32).reshape(num_moves, move_dim).copy()
    moves = np.ascontiguousarray(moves)   # ensure writable copy



    return moves, {
        'num_moves':       num_moves,
        'scalar_dim':      scalar_dim,
        'trunk_dim':       trunk_dim,
        'input_dim':       move_dim,
        'mean_log_prior':  summary[_SUM_MEAN_LP],
        'human_rank_idx':  int(summary[_SUM_HUMAN_RANK]) if summary[_SUM_HUMAN_RANK] >= 0 else -1,
        'human_log_prior': summary[_SUM_HUMAN_LP],
        'summary':         summary,
    }


# ─── Input type ──────────────────────────────────────────────────────────────

class SgfInput:
    """
    Unified SGF input container. Accepts any combination of:
      - paths:      list of .sgf file paths
      - strings:    list of SGF content strings (written to temp files)
      - directory:  path to a directory scanned for .sgf files
      - csv:        path to a pre-made CSV list file

    Call resolve() to get a (csv_path, cleanup_ctx) pair ready for batch_analysis.
    """

    def __init__(
        self,
        sgf_paths: Optional[Union[str, Iterable[str]]] = None,
        sgf_strings: Optional[Iterable[str]] = None,
        sgf_dir: Optional[str] = None,
        csv: Optional[str] = None,
    ):
        self._paths    = [sgf_paths] if isinstance(sgf_paths, str) else (
                         list(sgf_paths) if sgf_paths else None)
        self._strings  = list(sgf_strings) if sgf_strings else None
        self._dir      = sgf_dir
        self._csv      = csv

        if not any([self._paths, self._strings, self._dir, self._csv]):
            raise ValueError(
                "Provide at least one of: sgf_paths, sgf_strings, sgf_dir, csv"
            )

    def resolve(self) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
        """
        Returns (csv_path, tmp_dir).
        tmp_dir must be kept alive until batch_analysis completes.
        """
        # Direct CSV pass-through (no temp files needed)
        if self._csv is not None:
            return self._csv, None

        tmp = tempfile.TemporaryDirectory()
        sgf_files = []

        # Collect from directory
        if self._dir is not None:
            d = Path(self._dir)
            if not d.is_dir():
                raise NotADirectoryError(f"sgf_dir not found: {self._dir}")
            for f in sorted(d.glob("*.sgf")):
                sgf_files.append(str(f.resolve()))

        # Collect from file paths
        if self._paths is not None:
            for p in self._paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"SGF not found: {p}")
                sgf_files.append(os.path.abspath(p))

        # Write SGF strings to temp files
        if self._strings is not None:
            for i, content in enumerate(self._strings):
                path = os.path.join(tmp.name, f"_string_{i:06d}.sgf")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                sgf_files.append(path)

        if not sgf_files:
            raise ValueError("No SGF files resolved from input")

        # Write CSV list
        csv_path = os.path.join(tmp.name, '_list.csv')
        with open(csv_path, 'w') as f:
            f.write('File,Player Black,Player White,Score,BlackRating,WhiteRating,Set\n')
            for p in sgf_files:
                f.write(f'{p},unknown,unknown,0.5,1500,1500,T\n')

        return csv_path, tmp


# ─── Engine ───────────────────────────────────────────────────────────────────

class KataGoEngine:
    """
    Subprocess wrapper for `katago batch_analysis`.

    Locate the binary automatically (searches common paths) or pass it explicitly.
    """

    def __init__(
        self,
        model: str,
        config: Optional[str] = None,
        human_model: Optional[str] = None,
        katago_bin: Optional[str] = None,
    ):
        self.model       = str(model)
        self.config      = str(config) if config else None
        self.human_model = str(human_model) if human_model else None
        # discover_katago verifies fork capability (batch_analysis support)
        # even for explicit paths — stock katago fails fast here rather than
        # erroring mid-request.
        from katarank.katago_setup import discover_katago
        self.katago_bin  = discover_katago(katago_bin)

    def _base_cmd(self) -> List[str]:
        cmd = [self.katago_bin, 'batch_analysis', '-model', self.model]
        if self.config:
            cmd += ['-config', self.config]
        if self.human_model:
            cmd += ['-human-model', self.human_model]
        return cmd

    # ── Stream mode ───────────────────────────────────────────────────────────

    def stream_games(
        self,
        sgf_paths: Optional[Union[str, Iterable[str]]] = None,
        *,
        sgf_strings: Optional[Iterable[str]] = None,
        sgf_dir: Optional[str] = None,
        csv: Optional[str] = None,
        mode: str = 'full',
        min_moves: int = 10,
        max_games: int = 0,
    ) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
        """
        Run batch_analysis in stream mode; yield one record per player per game.

        Args (provide one):
            sgf_paths:   List of SGF file paths (or a single path).
            sgf_strings: List of SGF content strings.
            sgf_dir:     Directory to scan for .sgf files.
            csv:         Path to a pre-made CSV list file.

        Yields:
            (side, moves, info) per player per game.
            info['game_id'] carries the SGF filename stem from the stream frame.
        """
        inp = SgfInput(sgf_paths=sgf_paths, sgf_strings=sgf_strings, sgf_dir=sgf_dir, csv=csv)
        csv_path, tmp = inp.resolve()

        cmd = self._base_cmd() + [
            '-list', csv_path,
            '-min-moves', str(min_moves),
            '-stream',
        ]
        if mode == 'lite':
            cmd.append('-no-trunk')
        if max_games > 0:
            cmd += ['-max-games', str(max_games)]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=_CREATE_NO_WINDOW,
        )

        # Drain stderr in a background thread: katago writes progress there,
        # and an undrained pipe fills its buffer and deadlocks the process.
        stderr_tail: deque = deque(maxlen=50)
        def _drain():
            for line in iter(proc.stderr.readline, b''):
                stderr_tail.append(line.decode('utf-8', errors='replace'))
        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()

        try:
            yield from self._read_stream(proc.stdout)
        finally:
            proc.stdout.close()
            proc.wait()
            drainer.join(timeout=5)
            if tmp is not None:
                tmp.cleanup()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"katago batch_analysis exited with code {proc.returncode}\n"
                    + ''.join(stderr_tail)
                )

    @staticmethod
    def _read_exact(pipe, n: int) -> bytes:
        """Read exactly n bytes from pipe (or fewer at EOF)."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = pipe.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _read_stream(
        self, pipe, stop_at_job_marker: bool = False,
    ) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
        """Parse the game-id-tagged KAB2 stream from stdout pipe.

        Frame: [1B side][4B uint32 idLen][game id][4B uint32 size][payload].
        A single 0x00 byte terminates the stream; in daemon mode a 0x01
        byte marks the end of one job (set stop_at_job_marker=True).
        """
        while True:
            side_byte = self._read_exact(pipe, 1)
            if not side_byte or side_byte == _TERMINATOR:
                break
            if stop_at_job_marker and side_byte == _JOB_DONE:
                break
            side = chr(side_byte[0])

            id_len_raw = self._read_exact(pipe, 4)
            if len(id_len_raw) < 4:
                break
            id_len = struct.unpack('<I', id_len_raw)[0]
            game_id = self._read_exact(pipe, id_len).decode('utf-8', errors='replace')

            size_raw = self._read_exact(pipe, 4)
            if len(size_raw) < 4:
                break
            size = struct.unpack('<I', size_raw)[0]
            data = self._read_exact(pipe, size)
            if len(data) < size:
                break

            moves, info = parse_kab2_buffer(data)
            info['game_id'] = game_id
            yield side, moves, info

    # ── Ownership extraction (analysis JSON mode, stream path only) ──────────

    def fetch_ownership(
        self,
        sgf_content: str,
        *,
        max_visits: int = 1,
        timeout: int = 120,
        daemon: Optional['AnalysisDaemon'] = None,
    ) -> Dict[int, List[float]]:
        """Return per-position ownership for every turn.

        Prefers `daemon` (persistent AnalysisDaemon — no model reload) when
        provided. Falls back to a one-shot subprocess when daemon is absent.

        This is separate from batch_analysis (KAB2 format): KAB2 does not
        include ownership data. Only called in stream/online mode — never
        persisted.

        Args:
            sgf_content: Raw SGF string for a single game.
            max_visits:  NN visits per position (1 = single forward pass).
            timeout:     Timeout in seconds (subprocess or daemon query).
            daemon:      Optional persistent AnalysisDaemon; use when available.

        Returns:
            Dict of turn_number (0-based) → [361 floats].
            +1.0 = Black territory, -1.0 = White territory.
        """
        if daemon is not None and daemon.is_alive:
            return daemon.query_ownership(
                sgf_content, max_visits=max_visits, timeout=float(timeout)
            )

        # Fallback: one-shot subprocess (slow path — spawns and loads model)
        import json as _json
        from katarank._sgf_parse import extract_moves_for_analysis

        params = extract_moves_for_analysis(sgf_content)
        if params is None:
            return {}

        n_turns = len(params['moves']) + 1
        query = {
            'id': 'ownership-oneshot',
            'initialStones': [],
            'moves':         params['moves'],
            'rules':         params['rules'],
            'komi':          params['komi'],
            'boardXSize':    params['board_size'],
            'boardYSize':    params['board_size'],
            'maxVisits':     max_visits,
            'includeOwnership': True,
            'analyzeTurns':  list(range(n_turns)),
        }

        # `katago analysis` requires -config; auto-provision when absent
        from katarank.katago_setup import ensure_analysis_config
        cfg = ensure_analysis_config(self.config)
        cmd = [self.katago_bin, 'analysis', '-model', self.model,
               '-config', cfg]

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=_CREATE_NO_WINDOW,
            )
            stdout, _ = proc.communicate(
                input=(_json.dumps(query) + '\n').encode('utf-8'),
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return {}

        result: Dict[int, List[float]] = {}
        for line in stdout.decode('utf-8', errors='replace').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                if 'ownership' in obj and 'turnNumber' in obj:
                    result[obj['turnNumber']] = obj['ownership']
            except _json.JSONDecodeError:
                pass
        return result

    # ── File mode ─────────────────────────────────────────────────────────────

    def batch_to_files(
        self,
        output_dir: str,
        sgf_paths: Optional[Union[str, Iterable[str]]] = None,
        *,
        sgf_strings: Optional[Iterable[str]] = None,
        sgf_dir: Optional[str] = None,
        csv: Optional[str] = None,
        mode: str = 'full',
        min_moves: int = 10,
        max_games: int = 0,
    ) -> int:
        """
        Run batch_analysis writing one combined <stem>.npz per game to
        output_dir ([4B B_size][B KAB2][4B W_size][W KAB2], zlib compressed),
        plus a _meta.csv with per-game summaries.

        Args:
            output_dir:  Directory for output .npz files.
            sgf_paths:   List of SGF file paths.
            sgf_strings: List of SGF content strings.
            sgf_dir:     Directory to scan for .sgf files.
            csv:         Path to a pre-made CSV list file.

        Returns:
            Exit code (0 = success).
        """
        inp = SgfInput(sgf_paths=sgf_paths, sgf_strings=sgf_strings, sgf_dir=sgf_dir, csv=csv)
        csv_path, tmp = inp.resolve()
        os.makedirs(output_dir, exist_ok=True)

        cmd = self._base_cmd() + [
            '-list', csv_path,
            '-output-dir', output_dir,
            '-min-moves', str(min_moves),
        ]
        if mode == 'lite':
            cmd.append('-no-trunk')
        if max_games > 0:
            cmd += ['-max-games', str(max_games)]

        result = subprocess.run(cmd)
        if tmp is not None:
            tmp.cleanup()
        return result.returncode

    # ── Stream directly into model (convenience) ──────────────────────────────

    def stream_to_tensors(
        self,
        sgf_paths: Optional[Union[str, Iterable[str]]] = None,
        *,
        sgf_strings: Optional[Iterable[str]] = None,
        sgf_dir: Optional[str] = None,
        csv: Optional[str] = None,
        mode: str = 'full',
        device: str = 'cpu',
        **stream_kwargs,
    ):
        """
        Convenience: stream_games() → per-game (x_b, x_w, info_b, info_w).

        Frames are paired by game_id, so a game whose B or W side was
        dropped by katago (fewer than 5 moves) is skipped with a warning
        instead of silently corrupting the pairing.

        Yields torch tensors on `device`, ready for model.forward().
        """
        import torch
        import warnings

        pending: Dict[str, Tuple[str, 'torch.Tensor', Dict]] = {}
        gen = self.stream_games(
            sgf_paths, sgf_strings=sgf_strings, sgf_dir=sgf_dir, csv=csv,
            mode=mode, **stream_kwargs,
        )
        for side, moves, info in gen:
            x = torch.from_numpy(moves).to(device)
            gid = info.get('game_id', '')
            if gid not in pending:
                pending[gid] = (side, x, info)
                continue
            prev_side, prev_x, prev_info = pending.pop(gid)
            if prev_side == side:
                warnings.warn(f"Duplicate '{side}' frame for game {gid!r}; keeping latest")
                pending[gid] = (side, x, info)
                continue
            if side == 'W':
                yield prev_x, x, prev_info, info
            else:
                yield x, prev_x, info, prev_info

        for gid, (side, _, _) in pending.items():
            warnings.warn(
                f"Game {gid!r}: only side '{side}' received (other side <5 moves?) — skipped"
            )


# ─── Persistent engine (daemon mode) ─────────────────────────────────────────

class PersistentKataGoEngine(KataGoEngine):
    """
    Long-lived `katago batch_analysis -daemon` process.

    Model weights are loaded once at start(); each stream_games() call
    becomes a stdin job line instead of a fresh process, cutting per-call
    latency from ~tens of seconds (model load) to the analysis time itself.

    The analysis mode (lite/full) is fixed at start. Drop-in compatible with
    KataGoEngine.stream_games(); calls requesting a different mode fall back
    to a one-shot subprocess with a warning.

    Usage::

        with PersistentKataGoEngine(model='kata1.bin.gz', mode='lite') as eng:
            for side, moves, info in eng.stream_games(['g1.sgf']):
                ...
            for side, moves, info in eng.stream_games(['g2.sgf']):
                ...   # no model reload between calls
    """

    def __init__(
        self,
        model: str,
        config: Optional[str] = None,
        human_model: Optional[str] = None,
        katago_bin: Optional[str] = None,
        mode: str = 'lite',
        min_moves: int = 10,
    ):
        super().__init__(model=model, config=config,
                         human_model=human_model, katago_bin=katago_bin)
        self.mode = mode
        self.min_moves = min_moves
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_tail: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> 'PersistentKataGoEngine':
        """Spawn the daemon and wait until it reports ready."""
        if self._proc is not None:
            return self

        cmd = self._base_cmd() + ['-daemon', '-min-moves', str(self.min_moves)]
        if self.mode == 'lite':
            cmd.append('-no-trunk')

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=_CREATE_NO_WINDOW,
        )

        ready = threading.Event()
        def _drain():
            for line in iter(self._proc.stderr.readline, b''):
                text = line.decode('utf-8', errors='replace')
                self._stderr_tail.append(text)
                if 'daemon: ready' in text:
                    ready.set()
        self._drainer = threading.Thread(target=_drain, daemon=True)
        self._drainer.start()

        # Model load can take tens of seconds (GPU tuning on first run)
        while not ready.wait(timeout=1.0):
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"katago daemon exited with code {self._proc.returncode} "
                    f"during startup\n" + ''.join(self._stderr_tail)
                )
        return self

    def close(self, force: bool = False):
        """Stop the daemon.

        force=False — graceful: send 'quit', kill only on timeout.
        force=True  — kill immediately (wedged process / hung job).
        """
        if self._proc is None:
            return
        try:
            if force:
                self._proc.kill()
                self._proc.wait(timeout=5)
            elif self._proc.poll() is None:
                self._proc.stdin.write(b'quit\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()
        finally:
            self._proc = None

    def soft_reset(self):
        """Clear the daemon's NN caches and counters without reloading models.

        Cheap (~ms). Use between unrelated workloads or to free cache memory.
        Raises RuntimeError if the daemon is not running or died.
        """
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("daemon not running — call start() first")
        with self._lock:
            self._proc.stdin.write(b'reset\n')
            self._proc.stdin.flush()
            # The daemon acknowledges with the 0x01 job marker (no frames)
            for _ in self._read_stream(self._proc.stdout, stop_at_job_marker=True):
                pass

    def restart(self) -> 'PersistentKataGoEngine':
        """Hard reset: kill the process and start a fresh one (reloads models).

        Use when the daemon is wedged, leaks memory, or after a GPU error.
        """
        self.close(force=True)
        return self.start()

    def __enter__(self) -> 'PersistentKataGoEngine':
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ── job submission (same interface as KataGoEngine.stream_games) ─────────

    def stream_games(
        self,
        sgf_paths: Optional[Union[str, Iterable[str]]] = None,
        *,
        sgf_strings: Optional[Iterable[str]] = None,
        sgf_dir: Optional[str] = None,
        csv: Optional[str] = None,
        mode: Optional[str] = None,
        min_moves: Optional[int] = None,
        max_games: int = 0,
    ) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
        """Submit one job to the daemon and yield its frames.

        mode/min_moves are fixed at daemon start; a different requested mode
        falls back to a one-shot subprocess (slow path) with a warning.
        """
        if mode is not None and mode != self.mode:
            import warnings
            warnings.warn(
                f"PersistentKataGoEngine runs mode='{self.mode}'; request for "
                f"'{mode}' falls back to a one-shot subprocess"
            )
            yield from super().stream_games(
                sgf_paths, sgf_strings=sgf_strings, sgf_dir=sgf_dir, csv=csv,
                mode=mode, min_moves=min_moves or self.min_moves,
                max_games=max_games,
            )
            return

        if self._proc is None:
            self.start()

        inp = SgfInput(sgf_paths=sgf_paths, sgf_strings=sgf_strings,
                       sgf_dir=sgf_dir, csv=csv)
        csv_path, tmp = inp.resolve()

        # One job at a time: frames of concurrent jobs would interleave.
        with self._lock:
            try:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"katago daemon died (code {self._proc.returncode})\n"
                        + ''.join(self._stderr_tail)
                    )
                self._proc.stdin.write(csv_path.encode('utf-8') + b'\n')
                self._proc.stdin.flush()
                yield from self._read_stream(self._proc.stdout,
                                             stop_at_job_marker=True)
            finally:
                if tmp is not None:
                    tmp.cleanup()
