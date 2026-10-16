"""Failure-only counter for login lockout (per account key).

Separate from RateLimiter on purpose: RateLimiter counts every call, but a
lockout must count only FAILED attempts and reset on success.
Thread-safe, prunes stale keys so memory cannot grow without bound.
"""
import threading
import time
from typing import Callable, Dict, List


class FailureCounter:
    def __init__(self, max_failures: int, window_s: int,
                 clock: Callable[[], float] = time.time,
                 prune_above: int = 10_000) -> None:
        self._max = max_failures
        self._win = window_s
        self._clock = clock
        self._prune_above = prune_above
        self._fails: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _fresh(self, key: str, now: float) -> List[float]:
        return [t for t in self._fails.get(key, []) if now - t < self._win]

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._fresh(key, self._clock())) >= self._max

    def add(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            hits = self._fresh(key, now)
            hits.append(now)
            self._fails[key] = hits
            if len(self._fails) > self._prune_above:
                self._prune(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)

    def _prune(self, now: float) -> None:
        for k in list(self._fails):
            fresh = self._fresh(k, now)
            if fresh:
                self._fails[k] = fresh
            else:
                del self._fails[k]

    def __len__(self) -> int:
        with self._lock:
            return len(self._fails)