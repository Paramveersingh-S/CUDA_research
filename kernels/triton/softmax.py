import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def build_softmax_kernel(config):
    def kernel_fn(x, output, n_rows, n_cols):
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows, )
        softmax_kernel[grid](
            output, x, x.stride(0), output.stride(0), n_cols,
            num_warps=config.num_warps, BLOCK_SIZE=BLOCK_SIZE
        )
    return kernel_fn

def softmax_input_factory(shape):
    def factory():
        x = torch.randn(shape, device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)
        return (x, output, shape[0], shape[1])
    return factory
