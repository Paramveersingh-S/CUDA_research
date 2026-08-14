import pandas as pd
import os
import sys
import time
import triton

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.energy_autotuner import EnergyAutotuner
from kernels.triton.matmul import build_matmul_kernel, matmul_input_factory
from experiments.run_full_sweep import get_matmul_configs

RESULTS_PATH = "/content/drive/MyDrive/greentune_results/optuna_sweep.csv"

def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(RESULTS_PATH)
    df.to_csv(RESULTS_PATH, mode="a", header=header, index=False)

def run_optuna_sweep():
    print("--- Starting Phase 5: Optuna Search on GEMM ---")
    
    shape = (4096, 4096, 4096)
    configs = get_matmul_configs()
    
    print(f"Shape: {shape}, Total possible configurations in grid: {len(configs)}")
    print("Budget: We will give Optuna only 20 trials to find the Pareto optimal configs.")
    
    fac = matmul_input_factory(shape)
    
    tuner = EnergyAutotuner(
        kernel_builder=build_matmul_kernel,
        configs=configs,
        input_factory=fac
    )
    
    t0 = time.time()
    results = tuner.run_optuna(n_optuna_trials=20, n_trials=150)
    wall_clock_time = time.time() - t0
    
    print(f"\nOptuna search completed in {wall_clock_time:.2f} seconds.")
    
    # Save the configs that Optuna actually tested so we can compare
    for res in results:
        if "error" not in res:
            row = {
                "kernel": "matmul",
                "shape_str": str(shape),
                "config_str": str(res["config"]),
                "median_latency_s": res["median_latency_s"],
                "total_energy_j": res["total_energy_j"],
                "energy_per_call_j": res["energy_per_call_j"],
                "mean_watts": res["mean_watts"],
                "search_method": "optuna",
                "search_time_s": wall_clock_time
            }
            append_result(row)
            
    print(f"Saved {len([r for r in results if 'error' not in r])} successfully tested configs to {RESULTS_PATH}")
    print("Run `plot_search_comparison.py` to compare Optuna against the exhaustive Grid!")

if __name__ == "__main__":
    run_optuna_sweep()
