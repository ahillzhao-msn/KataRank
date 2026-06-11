"""
KataRank — KataGo Engine Interface
=====================================
Python wrapper around the `katago batch_analysis` binary.

Two analysis modes:
  'full'    — trunk + pick vectors, for training
  'lite'    — scalars only (10 dims), for fast inference

Two delivery modes:
  stream    — KAB2 frames via stdout pipe (zero disk I/O)
  file      — _B.npz/_W.npz files to output_dir

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
import tempfile
import zlib
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np


# ─── KAB2 constants ──────────────────────────────────────────────────────────

_MAGIC          = b'KAB2'
_HEADER_SIZE    = 96
_SUMMARY_OFFSET = 32

# PlayerSummary field indices
_SUM_MEAN_LP      = 2
_SUM_HUMAN_RANK   = 10
_SUM_HUMAN_LP     = 11

# Stream protocol constants
_FRAME_HEADER  = 5   # 1 byte side + 4 bytes uint32 size
_TERMINATOR    = b'\x00'


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
        self.model       = str(model)
        self.config      = str(config) if config else None
        self.human_model = str(human_model) if human_model else None
        self.katago_bin  = str(katago_bin) if katago_bin else self._find_binary()

    def _find_binary(self) -> str:
        import shutil
        for p in self._DEFAULT_BINS:
            if p.exists():
                return str(p)
        found = shutil.which('katago')
        if found:
            return found
        raise FileNotFoundError(
            "katago binary not found.\n"
            "  Option A: place katago.exe under  <project>/bin/katago.exe\n"
            "  Option B: add katago.exe to system PATH\n"
            "  Option C: pass katago_bin='/path/to/katago.exe' explicitly\n"
            "Download from: https://github.com/ahillzhao-msn/KataGo/releases"
        )

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
        )

        try:
            yield from self._read_stream(proc.stdout)
        finally:
            proc.stdout.close()
            proc.wait()
            if tmp is not None:
                tmp.cleanup()

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
        Run batch_analysis writing _B.npz/_W.npz files to output_dir.

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

        Yields torch tensors on `device`, ready for model.forward().
        """
        import torch

        buf_b: Optional[Tuple] = None
        gen = self.stream_games(
            sgf_paths, sgf_strings=sgf_strings, sgf_dir=sgf_dir, csv=csv,
            mode=mode, **stream_kwargs,
        )
        for side, moves, info in gen:
            x = torch.from_numpy(moves).to(device)
            if side == 'B':
                buf_b = (x, info)
            elif side == 'W' and buf_b is not None:
                yield buf_b[0], x, buf_b[1], info
                buf_b = None
