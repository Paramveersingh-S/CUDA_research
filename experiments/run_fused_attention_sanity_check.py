import sys
import os
import torch
import triton
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.energy_autotuner import EnergyAutotuner
from kernels.triton.fused_attention import build_fused_attention_kernel, fused_attention_input_factory

def run_attention_sanity():
    print("--- Phase 2: Sanity Check on Fused Causal Attention ---")
    shape = (2, 8, 1024, 64) # Z, H, N_CTX, D_HEAD
    
    print("Checking numerical correctness...")
    q, k, v, sm_scale = fused_attention_input_factory(shape)()
    
    # PyTorch baseline (SDPA)
    torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    config = triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3)
    triton_fn = build_fused_attention_kernel(config)
    triton_out = triton_fn(q, k, v, sm_scale)
    
    triton_err = torch.max(torch.abs(torch_out - triton_out)).item()
    print("Triton Error:", triton_err)
    assert triton_err < 1e-1, "Triton kernel differs from PyTorch SDPA"
    
    print("Correctness passed!")

    print("Running EnergyAutotuner Sweep (Fused Attention)...")
    configs = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]
    
    tuner = EnergyAutotuner(
        kernel_builder=build_fused_attention_kernel,
        configs=configs,
        input_factory=fused_attention_input_factory(shape)
    )
    
    results = tuner.run_grid(n_trials=100)
    for res in results:
        print(f"Config: {res['config']}, Latency: {res['median_latency_s']*1000:.3f} ms, Energy: {res['energy_per_call_j']*1000:.3f} mJ")
        
    print("Fused Attention Sanity Check Complete!")

if __name__ == "__main__":
    run_attention_sanity()
