import numpy as np


class RAGPerformanceMonitor:
    def __init__(self) -> None:
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "retrieval_ms": [],
            "ttfa_ms": [],
            "total_ms": [],
        }

    def log_retrieval(self, is_hit: bool, ms: float) -> None:
        key = "cache_hits" if is_hit else "cache_misses"
        self.metrics[key] += 1
        self.metrics["retrieval_ms"].append(ms)

    def log_ttfa(self, ms: float) -> None:
        self.metrics["ttfa_ms"].append(ms)

    def log_total(self, ms: float) -> None:
        self.metrics["total_ms"].append(ms)

    def summary(self) -> dict[str, str]:
        total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = (self.metrics["cache_hits"] / total * 100) if total > 0 else 0
        retrieval_ms = self.metrics["retrieval_ms"]
        ttfa_ms = self.metrics["ttfa_ms"]
        return {
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "avg_retrieval": f"{np.mean(retrieval_ms):.1f}ms" if retrieval_ms else "N/A",
            "p95_retrieval": (
                f"{np.percentile(retrieval_ms, 95):.1f}ms" if retrieval_ms else "N/A"
            ),
            "avg_ttfa": f"{np.mean(ttfa_ms):.0f}ms" if ttfa_ms else "N/A",
        }
