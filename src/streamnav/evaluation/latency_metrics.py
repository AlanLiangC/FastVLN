import numpy as np


def latency_summary(seconds):
    if not seconds:
        return {}
    p50, p95, p99 = np.percentile(seconds, [50, 95, 99]) * 1000
    return {
        "latency_p50_ms": float(p50),
        "latency_p95_ms": float(p95),
        "latency_p99_ms": float(p99),
        "decision_hz": float(1.0 / np.mean(seconds)),
    }
