import sys
import os
import torch
import triton

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.energy_autotuner import EnergyAutotuner
from kernels.triton.matmul import build_matmul_kernel, matmul_input_factory
from kernels.cuda.matmul_wrapper import matmul as cuda_matmul

def run_matmul_sanity():
    print("--- Phase 2: Sanity Check on Tiled GEMM (FP16) ---")
    shape = (2048, 2048, 2048) # M, N, K
    
    print("Checking numerical correctness...")
    a = torch.randn((shape[0], shape[2]), device='cuda', dtype=torch.float16)
    b = torch.randn((shape[2], shape[1]), device='cuda', dtype=torch.float16)
    
    torch_out = torch.matmul(a, b)
    
    cuda_out = cuda_matmul(a, b)
    # Tiled fp16 accumulation has higher tolerance than fp32
    cuda_err = torch.max(torch.abs(torch_out - cuda_out)).item()
    print("CUDA Error:", cuda_err)
    assert cuda_err < 1.0, "CUDA baseline differs from PyTorch cuBLAS"
    
    config = triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3)
    triton_fn = build_matmul_kernel(config)
    triton_out = torch.empty_like(torch_out)
    triton_fn(a, b, triton_out, shape[0], shape[1], shape[2])
    
    triton_err = torch.max(torch.abs(torch_out - triton_out)).item()
    print("Triton Error:", triton_err)
    assert triton_err < 1.0, "Triton kernel differs from PyTorch cuBLAS"
    
    print("Correctness passed!")

    print("Running EnergyAutotuner Sweep (GEMM)...")
    configs = [
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
    ]
    
    tuner = EnergyAutotuner(
        kernel_builder=build_matmul_kernel,
        configs=configs,
        input_factory=matmul_input_factory(shape)
    )
    
    results = tuner.run_grid(n_trials=100) # GEMM is fast enough but complex, 100 trials ~ hundreds of ms
    for res in results:
        print(f"Config: {res['config']}, Latency: {res['median_latency_s']*1000:.3f} ms, Energy: {res['energy_per_call_j']*1000:.3f} mJ")
        
    print("GEMM Sanity Check Complete!")

if __name__ == "__main__":
    run_matmul_sanity()
