"""Simple in-memory sliding-window rate limiter."""
import time
from typing import Dict, List


class RateLimiter:
    def __init__(self, max_calls: int, window_s: int) -> None:
        self._max = max_calls
        self._win = window_s
        self._hits: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        hits = [t for t in self._hits.get(key, []) if now - t < self._win]
        self._hits[key] = hits
        if len(hits) >= self._max:
            return False
        self._hits[key].append(now)
        return True


upload_limiter = RateLimiter(max_calls=20, window_s=60)
job_limiter    = RateLimiter(max_calls=10, window_s=60)
auth_limiter   = RateLimiter(max_calls=10, window_s=300)  # 10 login attempts / 5 min
chat_limiter   = RateLimiter(max_calls=20, window_s=60)   # 20 chat messages / min