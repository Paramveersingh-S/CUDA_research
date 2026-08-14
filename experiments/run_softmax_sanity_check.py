import sys
import os
import torch
import triton
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.energy_autotuner import EnergyAutotuner
from kernels.triton.softmax import build_softmax_kernel, softmax_input_factory
from kernels.cuda.softmax_wrapper import softmax as cuda_softmax

def run_softmax_sanity():
    print("--- Phase 2: Sanity Check on Fused Softmax ---")
    shape = (4096, 4096)
    
    print("Checking numerical correctness...")
    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    torch_out = F.softmax(x, dim=1)
    
    cuda_out = cuda_softmax(x)
    print("CUDA Error:", torch.max(torch.abs(torch_out - cuda_out)).item())
    assert torch.allclose(torch_out, cuda_out, atol=1e-3), "CUDA baseline differs from PyTorch"
    
    config = triton.Config({}, num_warps=4)
    triton_fn = build_softmax_kernel(config)
    triton_out = torch.empty_like(x)
    triton_fn(x, triton_out, shape[0], shape[1])
    print("Triton Error:", torch.max(torch.abs(torch_out - triton_out)).item())
    assert torch.allclose(torch_out, triton_out, atol=1e-3), "Triton kernel differs from PyTorch"
    
    print("Correctness passed!")

    print("Running EnergyAutotuner Sweep (Softmax)...")
    configs = [
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ]
    
    tuner = EnergyAutotuner(
        kernel_builder=build_softmax_kernel,
        configs=configs,
        input_factory=softmax_input_factory(shape)
    )
    
    results = tuner.run_grid(n_trials=200) # Softmax is slower, 200 is enough
    for res in results:
        print(f"Config: {res['config']}, Latency: {res['median_latency_s']*1000:.3f} ms, Energy: {res['energy_per_call_j']*1000:.3f} mJ")
        
    print("Softmax Sanity Check Complete!")

if __name__ == "__main__":
    run_softmax_sanity()
