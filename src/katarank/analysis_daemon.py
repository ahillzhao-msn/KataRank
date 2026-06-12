"""
AnalysisDaemon — persistent katago analysis subprocess.

Wraps `katago analysis` as a long-lived process using KataGo's native
JSON analysis protocol (stdin → stdout, one JSON line per direction).
Multiple callers can submit queries concurrently: each query is assigned
a UUID 'id'; a background reader thread routes responses back to the
correct caller via _QueryFuture.

KataGo returns one JSON response line per analyzeTurns entry. For a game
with N moves, a full-game ownership query produces N+1 responses (turn 0 =
empty board, turn N = final position).

Two analysis paths in katarank:
  PersistentKataGoEngine  →  katago batch_analysis -daemon  (KAB2 binary)
  AnalysisDaemon          →  katago analysis                (JSON protocol)

The two daemons run as independent processes to prevent interactive queries
from blocking behind long batch jobs on the GPU.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import uuid
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class _QueryFuture:
    """Accumulates N responses for one query, signals when complete."""
    __slots__ = ('expected', 'responses', '_event', '_exc')

    def __init__(self, expected: int) -> None:
        self.expected  = expected
        self.responses: List[dict] = []
        self._event    = threading.Event()
        self._exc: Optional[Exception] = None

    def add(self, response: dict) -> None:
        self.responses.append(response)
        if len(self.responses) >= self.expected:
            self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout=timeout)

    def fail(self, exc: Exception) -> None:
        self._exc = exc
        self._event.set()

    def result(self) -> List[dict]:
        if self._exc is not None:
            raise self._exc
        return self.responses


class AnalysisDaemon:
    """Persistent `katago analysis` subprocess for interactive/online queries.

    Model weights are loaded once at start(). Each query_*() call sends a
    JSON query on stdin and blocks until all expected responses arrive on
    stdout — no subprocess spawn per request.

    Usage::

        daemon = AnalysisDaemon(katago_bin, model, config)
        daemon.start()
        ownership = daemon.query_ownership(sgf_content)
        daemon.stop()

    Or as a context manager::

        with AnalysisDaemon(katago_bin, model, config) as d:
            ownership = d.query_ownership(sgf_content)
    """

    def __init__(
        self,
        katago_bin: str,
        model: str,
        config: Optional[str] = None,
    ) -> None:
        self._bin    = katago_bin
        self._model  = model
        self._config = config

        self._proc: Optional[subprocess.Popen] = None
        self._write_lock   = threading.Lock()   # serialise stdin writes
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, _QueryFuture] = {}
        self._stderr_tail: deque = deque(maxlen=200)
        self._alive = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> 'AnalysisDaemon':
        if self._alive:
            return self

        cmd = [self._bin, 'analysis', '-model', self._model]
        if self._config:
            cmd += ['-config', self._config]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._alive = True

        threading.Thread(target=self._drain_stderr, daemon=True,
                         name='katago-analysis-stderr').start()
        threading.Thread(target=self._reader_loop,  daemon=True,
                         name='katago-analysis-stdout').start()

        logger.info('AnalysisDaemon started (model=%s)', self._model)
        return self

    def stop(self) -> None:
        if not self._alive or self._proc is None:
            return
        self._alive = False
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()

        with self._pending_lock:
            for fut in self._pending.values():
                fut.fail(RuntimeError('AnalysisDaemon stopped'))
            self._pending.clear()
        self._proc = None
        logger.info('AnalysisDaemon stopped')

    @property
    def is_alive(self) -> bool:
        return self._alive and self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> 'AnalysisDaemon':
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Background threads ────────────────────────────────────────────────────

    def _drain_stderr(self) -> None:
        assert self._proc is not None
        for line in iter(self._proc.stderr.readline, b''):
            text = line.decode('utf-8', errors='replace').rstrip()
            self._stderr_tail.append(text)
            if text:
                logger.debug('[katago analysis] %s', text)

    def _reader_loop(self) -> None:
        assert self._proc is not None
        for raw in iter(self._proc.stdout.readline, b''):
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning('AnalysisDaemon malformed JSON: %.120s', line)
                continue

            qid = str(obj.get('id', ''))
            with self._pending_lock:
                fut = self._pending.get(qid)
            if fut is None:
                logger.debug('AnalysisDaemon orphan response id=%r', qid)
                continue
            fut.add(obj)

        # Process exited — fail all pending futures
        tail = '\n'.join(list(self._stderr_tail)[-15:])
        rc   = self._proc.returncode if self._proc else '?'
        exc  = RuntimeError(
            f'AnalysisDaemon process exited (code {rc})'
            + (f'\n{tail}' if tail else '')
        )
        with self._pending_lock:
            for fut in self._pending.values():
                fut.fail(exc)
            self._pending.clear()
        self._alive = False

    # ── Internal query plumbing ───────────────────────────────────────────────

    def _send(self, query: dict, expected: int, timeout: float) -> List[dict]:
        """Write one JSON query to stdin; block until `expected` responses arrive."""
        if not self._alive or self._proc is None:
            raise RuntimeError('AnalysisDaemon not running — call start() first')
        if self._proc.poll() is not None:
            raise RuntimeError(
                f'AnalysisDaemon process is dead (returncode={self._proc.returncode})'
            )

        qid = str(uuid.uuid4())
        fut = _QueryFuture(expected)

        with self._pending_lock:
            self._pending[qid] = fut

        with self._write_lock:
            payload = (json.dumps({**query, 'id': qid}) + '\n').encode('utf-8')
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()

        if not fut.wait(timeout=timeout):
            with self._pending_lock:
                self._pending.pop(qid, None)
            got = len(fut.responses)
            raise TimeoutError(
                f'AnalysisDaemon: query {qid[:8]}… timed out after {timeout}s '
                f'({got}/{expected} responses received)'
            )

        with self._pending_lock:
            self._pending.pop(qid, None)
        return fut.result()

    # ── Public query API ──────────────────────────────────────────────────────

    def query_ownership(
        self,
        sgf_content: str,
        *,
        max_visits: int = 1,
        timeout: float = 180.0,
    ) -> Dict[int, List[float]]:
        """Per-position ownership for every turn of a game.

        Args:
            sgf_content: Raw SGF string (single game).
            max_visits:  NN visits per position. 1 = single forward pass,
                         sufficient for territory overlay (~50ms/turn on GPU).
            timeout:     Total seconds to wait for all N+1 responses.

        Returns:
            Dict of turn_number (0-based) → [361 floats].
            Positive = Black territory, negative = White territory.
        """
        from katarank._sgf_parse import extract_moves_for_analysis

        params = extract_moves_for_analysis(sgf_content)
        if params is None:
            return {}

        n_turns = len(params['moves']) + 1   # include empty-board turn 0
        query = {
            'initialStones':    [],
            'moves':            params['moves'],
            'rules':            params['rules'],
            'komi':             params['komi'],
            'boardXSize':       params['board_size'],
            'boardYSize':       params['board_size'],
            'maxVisits':        max_visits,
            'includeOwnership': True,
            'analyzeTurns':     list(range(n_turns)),
        }

        responses = self._send(query, expected=n_turns, timeout=timeout)
        return {
            resp['turnNumber']: resp['ownership']
            for resp in responses
            if 'turnNumber' in resp and 'ownership' in resp
        }

    def query_variation(
        self,
        sgf_content: str,
        turn: int,
        *,
        max_visits: int = 50,
        include_policy: bool = True,
        timeout: float = 60.0,
    ) -> Optional[dict]:
        """Top moves and policy for a single turn (future: what-if branches).

        Args:
            sgf_content:    Raw SGF string.
            turn:           0-based turn number to analyse.
            max_visits:     NN visits (higher = stronger, slower).
            include_policy: Include per-position policy prior.
            timeout:        Seconds to wait for the single response.

        Returns:
            Raw KataGo analysis JSON object for that turn, or None on error.
        """
        from katarank._sgf_parse import extract_moves_for_analysis

        params = extract_moves_for_analysis(sgf_content)
        if params is None:
            return None
        if turn < 0 or turn > len(params['moves']):
            return None

        query = {
            'initialStones':  [],
            'moves':          params['moves'],
            'rules':          params['rules'],
            'komi':           params['komi'],
            'boardXSize':     params['board_size'],
            'boardYSize':     params['board_size'],
            'maxVisits':      max_visits,
            'includePolicy':  include_policy,
            'analyzeTurns':   [turn],
        }
        responses = self._send(query, expected=1, timeout=timeout)
        return responses[0] if responses else None
