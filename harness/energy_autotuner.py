import numpy as np
from .benchmark import benchmark_config

class EnergyAwareResult:
    def __init__(self, config, latency_s, energy_j):
        self.config = config
        self.latency_s = latency_s
        self.energy_j = energy_j

class EnergyAutotuner:
    """Grid- or Optuna-driven search over a Triton kernel's config space that
    records BOTH latency and energy per config, then exposes:
      - the latency-optimal config (== what triton.autotune would have picked)
      - the energy-optimal config
      - the full Pareto front
      - a scalarized 'knee' pick for a chosen tradeoff weight alpha
    """
    def __init__(self, kernel_builder, configs, input_factory):
        """kernel_builder(config) -> callable kernel_fn(*args)
        configs: list of dict-like Triton Config objects to sweep
        input_factory() -> args tuple for the given problem size"""
        self.kernel_builder = kernel_builder
        self.configs = configs
        self.input_factory = input_factory
        self.results = []

    def run_grid(self, n_trials=50):
        args = self.input_factory()
        for cfg in self.configs:
            fn = self.kernel_builder(cfg)
            try:
                stats = benchmark_config(fn, *args, n_trials=n_trials)
            except Exception as e:
                # Some configs are illegal (e.g. shared mem overflow) -- record and skip
                self.results.append(dict(config=cfg, error=str(e)))
                continue
            self.results.append(dict(config=cfg, **stats))
        return self.results

    def run_optuna(self, n_optuna_trials=20, n_trials=50):
        import optuna
        
        args = self.input_factory()
        
        def objective(trial):
            idx = trial.suggest_int('config_idx', 0, len(self.configs) - 1)
            cfg = self.configs[idx]
            fn = self.kernel_builder(cfg)
            
            try:
                stats = benchmark_config(fn, *args, n_trials=n_trials)
                self.results.append(dict(config=cfg, **stats))
                return stats['median_latency_s'], stats['energy_per_call_j']
            except Exception as e:
                self.results.append(dict(config=cfg, error=str(e)))
                return float('inf'), float('inf')
                
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(objective, n_trials=n_optuna_trials)
        
        return self.results

    def pareto_front(self):
        """Non-dominated set on (latency, energy), both minimized."""
        valid = [r for r in self.results if "error" not in r]
        front = []
        for r in valid:
            dominated = any(
                (o["median_latency_s"] <= r["median_latency_s"] and
                 o["energy_per_call_j"] <= r["energy_per_call_j"] and
                 (o["median_latency_s"] < r["median_latency_s"] or
                  o["energy_per_call_j"] < r["energy_per_call_j"]))
                for o in valid if o is not r
            )
            if not dominated:
                front.append(r)
        return sorted(front, key=lambda r: r["median_latency_s"])

    def scalarized_best(self, alpha=0.5):
        """alpha=1 -> pure latency optimizer (matches triton.autotune baseline)
        alpha=0 -> pure energy optimizer. Normalize both axes to [0,1] first."""
        valid = [r for r in self.results if "error" not in r]
        lat = np.array([r["median_latency_s"] for r in valid])
        en = np.array([r["energy_per_call_j"] for r in valid])
        lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-12)
        en_n = (en - en.min()) / (en.max() - en.min() + 1e-12)
        score = alpha * lat_n + (1 - alpha) * en_n
        return valid[int(np.argmin(score))]
