import sys
import os
import torch
import triton

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.energy_autotuner import EnergyAutotuner
from kernels.triton.rmsnorm import build_rmsnorm_kernel, rmsnorm_input_factory
from kernels.cuda.rmsnorm_wrapper import rmsnorm as cuda_rmsnorm

def pytorch_rmsnorm(x, weight, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

def run_rmsnorm_sanity():
    print("--- Phase 2: Sanity Check on RMSNorm ---")
    shape = (4096, 4096)
    eps = 1e-5
    
    print("Checking numerical correctness...")
    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    weight = torch.ones(shape[1], device='cuda', dtype=torch.float32)
    
    torch_out = pytorch_rmsnorm(x, weight, eps)
    
    cuda_out = cuda_rmsnorm(x, weight, eps)
    print("CUDA Error:", torch.max(torch.abs(torch_out - cuda_out)).item())
    assert torch.allclose(torch_out, cuda_out, atol=1e-3), "CUDA baseline differs from PyTorch"
    
    config = triton.Config({}, num_warps=4)
    triton_fn = build_rmsnorm_kernel(config)
    triton_out = torch.empty_like(x)
    triton_fn(x, weight, triton_out, shape[0], shape[1], eps=eps)
    print("Triton Error:", torch.max(torch.abs(torch_out - triton_out)).item())
    assert torch.allclose(torch_out, triton_out, atol=1e-3), "Triton kernel differs from PyTorch"
    
    print("Correctness passed!")

    print("Running EnergyAutotuner Sweep (RMSNorm)...")
    configs = [
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ]
    
    tuner = EnergyAutotuner(
        kernel_builder=build_rmsnorm_kernel,
        configs=configs,
        input_factory=rmsnorm_input_factory(shape)
    )
    
    results = tuner.run_grid(n_trials=200)
    for res in results:
        print(f"Config: {res['config']}, Latency: {res['median_latency_s']*1000:.3f} ms, Energy: {res['energy_per_call_j']*1000:.3f} mJ")
        
    print("RMSNorm Sanity Check Complete!")

if __name__ == "__main__":
    run_rmsnorm_sanity()
