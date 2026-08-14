import torch
import time
import numpy as np
from .power_sampler import PowerSampler

def benchmark_config(kernel_fn, *args, n_warmup=10, n_trials=50, **kwargs):
    """Runs a kernel n_trials times, measuring wall latency per call AND
    integrated energy over the whole trial block, discarding throttled trials."""
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        kernel_fn(*args, **kwargs)
    torch.cuda.synchronize()

    latencies = []
    with PowerSampler() as sampler:
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            kernel_fn(*args, **kwargs)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    total_energy_j = sampler.energy_joules()
    total_time_s = sum(latencies)
    energy_per_call_j = total_energy_j * (sum(latencies) / total_time_s) / n_trials \
        if total_time_s > 0 else float("nan")
    return dict(
        latencies_s=latencies,
        median_latency_s=float(np.median(latencies)),
        p95_latency_s=float(np.percentile(latencies, 95)),
        total_energy_j=total_energy_j,
        energy_per_call_j=energy_per_call_j,
        mean_watts=sampler.mean_watts(),
        n_trials=n_trials,
    )
