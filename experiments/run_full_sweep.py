import pandas as pd
import os
import torch
import triton
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.energy_autotuner import EnergyAutotuner

# Import builders and factories
from kernels.triton.vector_add import build_vector_add_kernel, vector_add_input_factory
from kernels.triton.softmax import build_softmax_kernel, softmax_input_factory
from kernels.triton.rmsnorm import build_rmsnorm_kernel, rmsnorm_input_factory
from kernels.triton.matmul import build_matmul_kernel, matmul_input_factory
from kernels.triton.fused_attention import build_fused_attention_kernel, fused_attention_input_factory

RESULTS_PATH = "/content/drive/MyDrive/greentune_results/full_sweep.csv"

def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(RESULTS_PATH)
    df.to_csv(RESULTS_PATH, mode="a", header=header, index=False)

def already_done(kernel, shape, config_str):
    if not os.path.exists(RESULTS_PATH):
        return False
    df = pd.read_csv(RESULTS_PATH)
    return ((df.kernel == kernel) & (df.shape_str == shape) & (df.config_str == config_str)).any()

# Define shapes
shapes = {
    "vector_add": [1024*1024*5, 1024*1024*10, 1024*1024*20], # S, M, L
    "softmax": [(2048, 2048), (4096, 4096), (8192, 8192)],
    "rmsnorm": [(2048, 2048), (4096, 4096), (8192, 8192)],
    "matmul": [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
    "fused_attention": [(2, 8, 512, 64), (2, 8, 1024, 64), (2, 8, 2048, 64)]
}

# Define config spaces
import itertools

def get_vector_add_configs():
    return [triton.Config({'BLOCK_SIZE': b}, num_warps=w) 
            for b in [256, 512, 1024, 2048] for w in [2, 4, 8, 16]]

def get_softmax_configs():
    return [triton.Config({}, num_warps=w) for w in [2, 4, 8, 16, 32]]

def get_rmsnorm_configs():
    return [triton.Config({}, num_warps=w) for w in [2, 4, 8, 16, 32]]

def get_matmul_configs():
    configs = []
    for block_m, block_n, block_k, group_m in [
        (32, 32, 32, 8), (64, 64, 32, 8), (128, 128, 32, 8),
        (64, 32, 32, 8), (32, 64, 32, 8), (128, 64, 32, 8)
    ]:
        for w in [2, 4, 8]:
            for stages in [2, 3, 4]:
                configs.append(triton.Config(
                    {'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_m},
                    num_warps=w, num_stages=stages
                ))
    return configs

def get_fused_attention_configs():
    configs = []
    for block_m, block_n in [(64, 32), (64, 64), (128, 64)]:
        for w in [4, 8]:
            for stages in [3, 4]:
                configs.append(triton.Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': block_n},
                    num_warps=w, num_stages=stages
                ))
    return configs

def run_sweep():
    tasks = [
        ("vector_add", build_vector_add_kernel, vector_add_input_factory, get_vector_add_configs()),
        ("softmax", build_softmax_kernel, softmax_input_factory, get_softmax_configs()),
        ("rmsnorm", build_rmsnorm_kernel, rmsnorm_input_factory, get_rmsnorm_configs()),
        ("matmul", build_matmul_kernel, matmul_input_factory, get_matmul_configs()),
        ("fused_attention", build_fused_attention_kernel, fused_attention_input_factory, get_fused_attention_configs())
    ]
    
    for kernel_name, builder, factory_fn, configs in tasks:
        print(f"--- Starting sweep for {kernel_name} ---")
        for shape in shapes[kernel_name]:
            shape_str = str(shape)
            print(f"  Shape: {shape_str}")
            
            fac = factory_fn(shape)
            
            for cfg in configs:
                config_str = str(cfg)
                if already_done(kernel_name, shape_str, config_str):
                    print(f"    Skipping {config_str} (already done)")
                    continue
                
                print(f"    Testing {config_str}...")
                
                tuner = EnergyAutotuner(
                    kernel_builder=builder,
                    configs=[cfg],
                    input_factory=fac
                )
                
                try:
                    results = tuner.run_grid(n_trials=150) # Balanced trials
                    if results and "error" not in results[0]:
                        res = results[0]
                        row = {
                            "kernel": kernel_name,
                            "shape_str": shape_str,
                            "config_str": config_str,
                            "median_latency_s": res["median_latency_s"],
                            "p95_latency_s": res["p95_latency_s"],
                            "total_energy_j": res["total_energy_j"],
                            "energy_per_call_j": res["energy_per_call_j"],
                            "mean_watts": res["mean_watts"]
                        }
                        append_result(row)
                    else:
                        print(f"      Error in config: {results[0].get('error') if results else 'Unknown'}")
                except Exception as e:
                    print(f"      Crashed: {e}")
                    
if __name__ == "__main__":
    run_sweep()
