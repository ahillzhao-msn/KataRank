"""
KataRank — KataGo Pipeline Pool

Manages a named collection of KataGo analysis pipelines.  Each pipeline
is one KataGoEngine + one StreamQueue.  The pool supports:

  - Starting / stopping individual pipelines
  - Iterating across all active queues (round-robin or priority)
  - Merging multiple live streams into a single iterator for training
  - Status reporting

Typical usage::

    pool = KataGoPool(model='kata1-b18c384nbt.bin.gz')

    # Add two pipelines: one for training data, one for online evaluation
    pool.start('train', sgfs=training_csv, mode='full', buffer_size=64)
    pool.start('eval',  sgfs=eval_csv,    mode='lite',  buffer_size=16)

    # Iterate merged stream (round-robin)
    for moves_b, moves_w, info_b, info_w in pool.merged():
        ...

    # Or access a specific queue
    for item in pool.queue('train'):
        ...

    pool.close_all()
"""

import threading
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from katarank.engine import KataGoEngine, StreamQueue


# ─── Pipeline record ──────────────────────────────────────────────────────────

class Pipeline:
    """One named analysis pipeline: engine + queue + metadata."""

    def __init__(
        self,
        name: str,
        engine: KataGoEngine,
        queue: StreamQueue,
        mode: str,
    ):
        self.name   = name
        self.engine = engine
        self.queue  = queue
        self.mode   = mode             # 'full' or 'lite'
        self.started_at = time.time()
        self.games_consumed = 0

    def status(self) -> dict:
        return {
            'name':    self.name,
            'mode':    self.mode,
            'alive':   not self.queue._done.is_set(),
            'buffered': self.queue._q.qsize(),
            'consumed': self.games_consumed,
            'uptime_s': round(time.time() - self.started_at, 1),
        }


# ─── Pool ─────────────────────────────────────────────────────────────────────

class KataGoPool:
    """
    Persistent manager for multiple KataGo analysis pipelines.

    Thread-safe: pipeline start/stop can be called from any thread.
    """

    def __init__(
        self,
        model: str,
        config: Optional[str]      = None,
        human_model: Optional[str] = None,
        katago_bin: Optional[str]  = None,
    ):
        """
        All pipelines share the same KataGo binary and model weights.

        Args:
            model:       Path to main KataGo model (.bin.gz)
            config:      KataGo config (.cfg); optional
            human_model: HumanSL model for rank annotation; optional
            katago_bin:  KataGo binary path; auto-detected if None
        """
        self._engine_kwargs = dict(
            model       = model,
            config      = config,
            human_model = human_model,
            katago_bin  = katago_bin,
        )
        self._pipelines: Dict[str, Pipeline] = {}
        self._lock = threading.Lock()

    # ── Pipeline lifecycle ────────────────────────────────────────────────────

    def start(
        self,
        name: str,
        sgfs: Union[Iterable[str], str],
        mode: str = 'full',
        min_moves: int = 10,
        max_games: int = 0,
        compress: bool = True,
        buffer_size: int = 32,
    ) -> StreamQueue:
        """
        Start a named pipeline and return its StreamQueue.

        If a pipeline with the same name already exists and is still alive,
        it is stopped first.

        Args:
            name:        Unique pipeline identifier.
            sgfs:        SGF paths, a CSV list file, or a Python iterable.
            mode:        'full' or 'lite'.
            buffer_size: Max paired games buffered in memory.
        """
        self.stop(name, wait=True)

        engine = KataGoEngine(**self._engine_kwargs)
        queue  = engine.stream_queue(
            sgfs,
            mode        = mode,
            min_moves   = min_moves,
            max_games   = max_games,
            compress    = compress,
            buffer_size = buffer_size,
        )
        pipe = Pipeline(name=name, engine=engine, queue=queue, mode=mode)

        with self._lock:
            self._pipelines[name] = pipe

        return queue

    def stop(self, name: str, wait: bool = False) -> None:
        """Stop and remove a named pipeline."""
        with self._lock:
            pipe = self._pipelines.pop(name, None)
        if pipe is not None:
            pipe.queue.close()
            if wait and pipe.queue._thread is not None:
                pipe.queue._thread.join(timeout=10.0)

    def close_all(self, wait: bool = True) -> None:
        """Stop all active pipelines."""
        names = list(self._pipelines.keys())
        for name in names:
            self.stop(name, wait=wait)

    # ── Data access ───────────────────────────────────────────────────────────

    def queue(self, name: str) -> StreamQueue:
        """Return the StreamQueue for a named pipeline."""
        with self._lock:
            if name not in self._pipelines:
                raise KeyError(f"No pipeline '{name}'. Active: {list(self._pipelines)}")
            return self._pipelines[name].queue

    def merged(
        self,
        names: Optional[List[str]] = None,
        strategy: str = 'roundrobin',
        timeout: float = 0.5,
    ) -> Iterator[Tuple]:
        """
        Merge active queues into a single iterator.

        Yields (moves_b, moves_w, info_b, info_w) from whichever queue
        has data.  Stops when ALL selected queues are exhausted.

        Args:
            names:    Queues to merge (None = all active pipelines).
            strategy: 'roundrobin' — rotate through queues in order.
                      'any'        — yield from whichever is ready first.
            timeout:  Per-queue .get() timeout in seconds.
        """
        if strategy == 'roundrobin':
            yield from self._merge_roundrobin(names, timeout)
        elif strategy == 'any':
            yield from self._merge_any(names, timeout)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'roundrobin' or 'any'.")

    def _active_queues(self, names: Optional[List[str]]) -> List[Tuple[str, StreamQueue]]:
        with self._lock:
            items = list(self._pipelines.items())
        if names is not None:
            items = [(n, p.queue) for n, p in self._pipelines.items() if n in names]
        else:
            items = [(n, p.queue) for n, p in items]
        return items

    def _merge_roundrobin(self, names, timeout) -> Iterator[Tuple]:
        queues = self._active_queues(names)
        exhausted = set()
        while len(exhausted) < len(queues):
            for name, q in queues:
                if name in exhausted:
                    continue
                item = q.get(timeout=timeout)
                if item is None:
                    exhausted.add(name)
                    # Record pipeline as done
                    with self._lock:
                        if name in self._pipelines:
                            self._pipelines[name].games_consumed += 0
                else:
                    with self._lock:
                        if name in self._pipelines:
                            self._pipelines[name].games_consumed += 1
                    yield item

    def _merge_any(self, names, timeout) -> Iterator[Tuple]:
        """Poll all queues; yield from the first one that has data."""
        import queue as _queue
        queues = self._active_queues(names)
        exhausted = set()
        while len(exhausted) < len(queues):
            found_any = False
            for name, q in queues:
                if name in exhausted:
                    continue
                try:
                    item = q._q.get_nowait()
                except _queue.Empty:
                    if q._done.is_set():
                        exhausted.add(name)
                    continue
                if item is q._q.__class__ or item is None:
                    exhausted.add(name)
                    continue
                from katarank.engine import _STREAM_DONE
                if item is _STREAM_DONE:
                    exhausted.add(name)
                    continue
                with self._lock:
                    if name in self._pipelines:
                        self._pipelines[name].games_consumed += 1
                yield item
                found_any = True
            if not found_any:
                time.sleep(0.01)

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> List[dict]:
        """Return status for all pipelines."""
        with self._lock:
            return [p.status() for p in self._pipelines.values()]

    def __repr__(self) -> str:
        s = self.status()
        lines = [f"KataGoPool ({len(s)} pipelines)"]
        for p in s:
            lines.append(
                f"  [{p['name']}] mode={p['mode']}  "
                f"alive={p['alive']}  buffered={p['buffered']}  "
                f"consumed={p['consumed']}  uptime={p['uptime_s']}s"
            )
        return '\n'.join(lines)
