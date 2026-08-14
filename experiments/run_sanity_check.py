import sys
import os
import torch
import triton

# Add root directory to path so we can import from harness and kernels
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.energy_autotuner import EnergyAutotuner
from kernels.triton.vector_add import build_vector_add_kernel, vector_add_input_factory
from kernels.cuda.vector_add_wrapper import add as cuda_add

def run_sanity_check():
    print("--- Phase 2: Sanity Check on Vector Add ---")
    size = 1024 * 1024 * 10  # 10M elements
    
    # 1. Numerical Correctness Check
    print("Checking numerical correctness...")
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    torch_out = x + y
    
    cuda_out = cuda_add(x, y)
    print("CUDA Error:", torch.max(torch.abs(torch_out - cuda_out)).item())
    assert torch.allclose(torch_out, cuda_out, atol=1e-3), "CUDA baseline differs from PyTorch"
    
    # Check Triton with one config
    config = triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4})
    triton_fn = build_vector_add_kernel(config)
    triton_out = torch.empty_like(x)
    triton_fn(x, y, triton_out, size)
    print("Triton Error:", torch.max(torch.abs(torch_out - triton_out)).item())
    assert torch.allclose(torch_out, triton_out, atol=1e-3), "Triton kernel differs from PyTorch"
    
    print("Correctness passed!")

    # 2. Autotuner and Harness Check
    print("Running EnergyAutotuner Sweep (Small Grid)...")
    configs = [
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}),
    ]
    
    tuner = EnergyAutotuner(
        kernel_builder=build_vector_add_kernel,
        configs=configs,
        input_factory=vector_add_input_factory(size)
    )
    
    results = tuner.run_grid(n_trials=30)
    for res in results:
        print(f"Config: {res['config']}, Latency: {res['median_latency_s']*1000:.3f} ms, Energy: {res['energy_per_call_j']*1000:.3f} mJ")
        
    print("Sanity Check Complete!")

if __name__ == "__main__":
    run_sanity_check()
