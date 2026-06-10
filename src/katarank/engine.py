"""
KataRank — KataGo Engine Interface
=====================================
Python wrapper around the `katago batch_analysis` binary.

Two output modes:
  'full'    — trunk + pick vectors (input_dim = 10 + 2*trunkCh)
              Used for training KataRankModel.
  'lite'    — scalars only (input_dim = 10, -no-trunk flag)
              Used for fast rank inference / evaluation without saving features.

Two delivery modes:
  stream    — KataGo pipes KAB2 frames directly to Python via stdout.
              No disk I/O; yields (side, moves_np, info) per player per game.
  file      — KataGo writes _B.npz/_W.npz files to output_dir (default behaviour).

Typical usage:

    # Stream directly into model (no disk write)
    engine = KataGoEngine(model='kata1-b18c384nbt.bin.gz')
    for side, moves, info in engine.stream_games(['game1.sgf', 'game2.sgf']):
        x = torch.from_numpy(moves).float()
        print(side, info['mean_log_prior'], info['human_rank_idx'])

    # Lite mode: scalars only, fast assessment
    for side, moves, info in engine.stream_games(sgfs, mode='lite'):
        pass   # moves.shape == (N, 10)

    # Traditional file-based batch
    engine.batch_to_files(sgfs, output_dir='features/', human_model='humansl.bin.gz')
"""

import os
import queue
import struct
import subprocess
import tempfile
import threading
import zlib
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np


# ─── KAB2 constants ──────────────────────────────────────────────────────────

_MAGIC          = b'KAB2'
_HEADER_SIZE    = 96   # sizeof(NPZHeader)
_SUMMARY_OFFSET = 32   # offset of PlayerSummary within header
_SUMMARY_FLOATS = 16

# PlayerSummary field indices
_SUM_MEAN_LP      = 2
_SUM_HUMAN_RANK   = 10
_SUM_HUMAN_LP     = 11

# Stream protocol constants
_FRAME_HEADER  = 5   # 1 byte side + 4 bytes uint32 size
_TERMINATOR    = b'\x00'


# ─── Low-level KAB2 parser (bytes → ndarray) ──────────────────────────────────

def parse_kab2_buffer(data: bytes) -> Tuple[np.ndarray, Dict]:
    """
    Parse a KAB2 payload from raw bytes (in-memory, no file I/O).

    Returns:
        moves:  (numMoves, input_dim) float32 array
        info:   dict with header fields and PlayerSummary values
    """
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
    moves = np.frombuffer(payload, dtype=np.float32).reshape(num_moves, move_dim)
    moves = np.ascontiguousarray(moves)   # ensure writable copy

    info = {
        'num_moves':       num_moves,
        'scalar_dim':      scalar_dim,
        'trunk_dim':       trunk_dim,
        'input_dim':       move_dim,
        'mean_log_prior':  summary[_SUM_MEAN_LP],
        'human_rank_idx':  int(summary[_SUM_HUMAN_RANK]) if summary[_SUM_HUMAN_RANK] >= 0 else -1,
        'human_log_prior': summary[_SUM_HUMAN_LP],
        'summary':         summary,
    }
    return moves, info


# ─── Engine ───────────────────────────────────────────────────────────────────

class KataGoEngine:
    """
    Subprocess wrapper for `katago batch_analysis`.

    Locate the binary automatically (searches common paths) or pass it explicitly.
    """

    # Paths searched in order when katago_bin is not given explicitly.
    # parents[0] = src/katarank/ — works both in dev (editable) and installed wheel.
    _DEFAULT_BINS = [
        Path(__file__).parent / 'bin' / 'katago.exe',
        Path(__file__).parent / 'bin' / 'katago',
    ]

    def __init__(
        self,
        model: str,
        config: Optional[str] = None,
        human_model: Optional[str] = None,
        katago_bin: Optional[str] = None,
    ):
        """
        Args:
            model:       Path to main KataGo model (.bin.gz)
            config:      Path to KataGo config (.cfg); optional
            human_model: Path to HumanSL model for rank annotation; optional
            katago_bin:  Explicit path to katago binary; auto-detected if None
        """
        self.model       = str(model)
        self.config      = str(config) if config else None
        self.human_model = str(human_model) if human_model else None
        self.katago_bin  = str(katago_bin) if katago_bin else self._find_binary()

    def _find_binary(self) -> str:
        import shutil
        # 1. Explicit paths relative to project root
        for p in self._DEFAULT_BINS:
            if p.exists():
                return str(p)
        # 2. System PATH
        found = shutil.which('katago')
        if found:
            return found
        raise FileNotFoundError(
            "katago binary not found.\n"
            "  Option A: place katago.exe under  <project>/bin/katago.exe\n"
            "  Option B: add katago.exe to your system PATH\n"
            "  Option C: pass  katago_bin='/path/to/katago.exe'  explicitly\n"
            "Download from: https://github.com/ahillzhao-msn/KataGo/releases"
        )

    def _base_cmd(self) -> List[str]:
        cmd = [self.katago_bin, 'batch_analysis', '-model', self.model]
        if self.config:
            cmd += ['-config', self.config]
        if self.human_model:
            cmd += ['-human-model', self.human_model]
        return cmd

    # ── Input helpers ─────────────────────────────────────────────────────────

    def _make_list_file(
        self, sgfs: Iterable[str], tmp_dir: str
    ) -> str:
        """Write a simple CSV list file for batch_analysis -list."""
        list_path = os.path.join(tmp_dir, '_input.csv')
        with open(list_path, 'w') as f:
            f.write('File,Player Black,Player White,BlackRating,WhiteRating,Set\n')
            for s in sgfs:
                f.write(f'{s},unknown,unknown,1500,1500,T\n')
        return list_path

    # ── Stream mode ───────────────────────────────────────────────────────────

    def stream_games(
        self,
        sgfs: Union[Iterable[str], str],
        mode: str = 'full',
        min_moves: int = 10,
        max_games: int = 0,
        compress: bool = True,
    ) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
        """
        Run batch_analysis in stream mode; yield one record per player per game.

        Yields:
            (side, moves, info)
            side:  'B' or 'W'
            moves: (N, input_dim) float32 numpy array
            info:  dict with header fields (mean_log_prior, human_rank_idx, …)

        Args:
            sgfs:      Iterable of SGF paths (or a single path / a CSV list path)
            mode:      'full' (trunk+scalars) or 'lite' (scalars only, 10 dims)
            min_moves: Minimum moves to include a game
            max_games: Cap number of games (0 = no cap)
            compress:  zlib-compress the KAB2 payloads (reduces pipe bandwidth)
        """
        with tempfile.TemporaryDirectory() as tmp:
            if isinstance(sgfs, str) and sgfs.endswith('.csv'):
                list_file = sgfs
            else:
                sgf_list = [sgfs] if isinstance(sgfs, str) else list(sgfs)
                list_file = self._make_list_file(sgf_list, tmp)

            cmd = self._base_cmd() + [
                '-list', list_file,
                '-min-moves', str(min_moves),
                '-stream',
            ]
            if mode == 'lite':
                cmd.append('-no-trunk')
            if not compress:
                cmd.append('-no-compress')
            if max_games > 0:
                cmd += ['-max-games', str(max_games)]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,   # KataGo prints progress to stderr
                bufsize=0,
            )

            try:
                yield from self._read_stream(proc.stdout)
            finally:
                proc.stdout.close()
                proc.wait()

    def _read_stream(self, pipe) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
        """Parse the length-prefixed KAB2 stream from stdout pipe."""
        while True:
            hdr = pipe.read(_FRAME_HEADER)
            if not hdr or len(hdr) < _FRAME_HEADER:
                break
            side_byte = hdr[0:1]
            if side_byte == _TERMINATOR:
                break
            side = chr(side_byte[0])
            size = struct.unpack_from('<I', hdr, 1)[0]
            data = pipe.read(size)
            if len(data) < size:
                break
            moves, info = parse_kab2_buffer(data)
            yield side, moves, info

    # ── File mode ─────────────────────────────────────────────────────────────

    def batch_to_files(
        self,
        sgfs: Union[Iterable[str], str],
        output_dir: str,
        mode: str = 'full',
        min_moves: int = 10,
        max_games: int = 0,
        compress: bool = True,
    ) -> int:
        """
        Run batch_analysis writing _B.npz/_W.npz files to output_dir.

        Returns:
            Exit code (0 = success)
        """
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            if isinstance(sgfs, str) and sgfs.endswith('.csv'):
                list_file = sgfs
            else:
                sgf_list = [sgfs] if isinstance(sgfs, str) else list(sgfs)
                list_file = self._make_list_file(sgf_list, tmp)

            cmd = self._base_cmd() + [
                '-list', list_file,
                '-output-dir', output_dir,
                '-min-moves', str(min_moves),
            ]
            if mode == 'lite':
                cmd.append('-no-trunk')
            if not compress:
                cmd.append('-no-compress')
            if max_games > 0:
                cmd += ['-max-games', str(max_games)]

            result = subprocess.run(cmd)
            return result.returncode

    # ── High-level: stream directly into model ────────────────────────────────

    def stream_to_tensors(
        self,
        sgfs: Union[Iterable[str], str],
        mode: str = 'full',
        device: str = 'cpu',
        **stream_kwargs,
    ):
        """
        Convenience wrapper: stream_games() → torch tensors ready for model.forward().

        Yields:
            (x_b, x_w, info_b, info_w)  per game
            x_b, x_w: (N, input_dim) float32 tensors on `device`
        """
        import torch

        buf_b: Optional[Tuple] = None
        for side, moves, info in self.stream_games(sgfs, mode=mode, **stream_kwargs):
            x = torch.from_numpy(moves).to(device)
            if side == 'B':
                buf_b = (x, info)
            elif side == 'W' and buf_b is not None:
                yield buf_b[0], x, buf_b[1], info
                buf_b = None

    # ── Queue mode (producer/consumer, for training loops) ────────────────────

    def stream_queue(
        self,
        sgfs: Union[Iterable[str], str],
        mode: str = 'full',
        min_moves: int = 10,
        max_games: int = 0,
        compress: bool = True,
        buffer_size: int = 32,
    ) -> 'StreamQueue':
        """
        Launch KataGo in a background thread and return a StreamQueue.

        The producer thread reads KAB2 frames, pairs B/W, and pushes
        (moves_b, moves_w, info_b, info_w) tuples to the queue.
        The caller iterates over the queue in its own (training) thread.

        Args:
            buffer_size: Max number of paired games held in memory.
                         The producer blocks when full, providing back-pressure.

        Example::

            sq = engine.stream_queue(sgfs, mode='full', buffer_size=64)
            for moves_b, moves_w, info_b, info_w in sq:
                batch = kab2_collate_arrays([(moves_b, moves_w, info_b, info_w)])
                train_step(batch)
        """
        q = StreamQueue(buffer_size=buffer_size)

        # Build list file; if we created a temp file we keep the TempDir alive
        # by attaching it to the queue so it's cleaned up only when the queue is done.
        if isinstance(sgfs, str) and sgfs.endswith('.csv'):
            tmp_dir = None
            list_file = sgfs
        else:
            tmp_dir = tempfile.TemporaryDirectory()
            sgf_list = [sgfs] if isinstance(sgfs, str) else list(sgfs)
            list_file = self._make_list_file(sgf_list, tmp_dir.name)

        q._tmp_dir = tmp_dir  # keep alive until queue is GC'd

        cmd = self._base_cmd() + [
            '-list', list_file,
            '-min-moves', str(min_moves),
            '-stream',
        ]
        if mode == 'lite':
            cmd.append('-no-trunk')
        if not compress:
            cmd.append('-no-compress')
        if max_games > 0:
            cmd += ['-max-games', str(max_games)]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        q._start(proc)
        return q


# ─── StreamQueue ─────────────────────────────────────────────────────────────

# Sentinel pushed by producer to signal end-of-stream
_STREAM_DONE = object()

# Type alias for one paired game sample
_GameSample = Tuple[np.ndarray, np.ndarray, Dict, Dict]


class StreamQueue:
    """
    Thread-safe bridge between a KataGo stdout pipe and a training loop.

    Architecture
    ------------
    Producer thread  →  queue.Queue  ←  consumer (training loop)

    The producer thread:
      - Reads raw KAB2 frames from the pipe (I/O-bound, releases GIL)
      - Parses them (CPU-bound but fast; numpy ops release GIL)
      - Pairs B/W frames; orphaned single-side frames are silently dropped
      - Pushes (moves_b, moves_w, info_b, info_w) to the queue
      - Blocks when the queue is full (back-pressure)
      - Pushes _STREAM_DONE sentinel when the pipe closes

    Consumer (training loop):
      - Iterates over the StreamQueue object (or calls .get())
      - Receives numpy arrays; shared by reference between threads (no copy)
      - torch.from_numpy() shares the same buffer (still no copy)

    Memory note
    -----------
    numpy arrays in the queue are heap-allocated Python objects.
    Threads share the same process heap; passing arrays through a
    queue.Queue is a reference transfer, not a copy.

    Back-pressure
    -------------
    buffer_size caps memory at roughly  buffer_size × moves_per_game × input_dim × 4 bytes.
    At buffer_size=32, full-mode (~778 dims, ~300 moves/player) ≈ 32 × 2 × 300 × 778 × 4 B ≈ 60 MB.
    Lite mode (10 dims) ≈ 0.8 MB at the same buffer_size.
    """

    def __init__(self, buffer_size: int = 32):
        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._error: Optional[Exception] = None
        self._done = threading.Event()
        self._tmp_dir: Optional[tempfile.TemporaryDirectory] = None  # set by KataGoEngine

    # ── Internal ──────────────────────────────────────────────────────────────

    def _start(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._thread = threading.Thread(
            target=self._produce,
            args=(proc.stdout,),
            daemon=True,
            name='katarank-stream-producer',
        )
        self._thread.start()

    def _produce(self, pipe) -> None:
        try:
            buf_b: Optional[Tuple[np.ndarray, Dict]] = None
            while True:
                hdr = pipe.read(_FRAME_HEADER)
                if not hdr or len(hdr) < _FRAME_HEADER:
                    break
                side_byte = hdr[0:1]
                if side_byte == _TERMINATOR:
                    break
                side = chr(side_byte[0])
                size = struct.unpack_from('<I', hdr, 1)[0]
                data = pipe.read(size)
                if len(data) < size:
                    break

                moves, info = parse_kab2_buffer(data)

                if side == 'B':
                    # If previous B was orphaned, silently discard it
                    buf_b = (moves, info)
                elif side == 'W':
                    if buf_b is not None:
                        self._q.put((buf_b[0], moves, buf_b[1], info))
                        buf_b = None
                    # W without preceding B: orphan, silently drop

        except Exception as exc:
            self._error = exc
        finally:
            pipe.close()
            if self._proc is not None:
                self._proc.wait()
            self._done.set()
            # Wake any blocked consumer
            try:
                self._q.put(_STREAM_DONE, timeout=2.0)
            except queue.Full:
                pass

    # ── Consumer interface ────────────────────────────────────────────────────

    def get(self, timeout: float = 1.0) -> Optional[_GameSample]:
        """
        Get the next paired (moves_b, moves_w, info_b, info_w) or None at end.

        Raises the producer's exception if one occurred.
        """
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                if self._done.is_set():
                    if self._error:
                        raise self._error
                    return None
                continue

            if item is _STREAM_DONE:
                if self._error:
                    raise self._error
                return None
            return item  # type: ignore[return-value]

    def __iter__(self) -> Iterator[_GameSample]:
        """Iterate until the stream ends or an error is raised."""
        while True:
            item = self.get()
            if item is None:
                return
            yield item

    def close(self) -> None:
        """Terminate the KataGo process and drain the queue."""
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    def __del__(self) -> None:
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
