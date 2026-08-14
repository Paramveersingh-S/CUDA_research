import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    output_ptr, input_ptr, weight_ptr, 
    input_row_stride, output_row_stride, 
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    
    # RMSNorm
    row_sq = row * row
    variance = tl.sum(row_sq, axis=0) / n_cols
    rsqrt = tl.math.rsqrt(variance + eps)
    
    output = row * rsqrt * weight
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

def build_rmsnorm_kernel(config):
    def kernel_fn(x, weight, output, n_rows, n_cols, eps=1e-5):
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows, )
        rmsnorm_kernel[grid](
            output, x, weight, 
            x.stride(0), output.stride(0), 
            n_cols, eps,
            num_warps=config.num_warps, BLOCK_SIZE=BLOCK_SIZE
        )
    return kernel_fn

def rmsnorm_input_factory(shape):
    def factory():
        x = torch.randn(shape, device='cuda', dtype=torch.float32)
        weight = torch.ones(shape[1], device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)
        return (x, weight, output, shape[0], shape[1])
    return factory
