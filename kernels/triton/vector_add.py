import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def build_vector_add_kernel(config):
    """Factory for the energy autotuner"""
    def kernel_fn(x, y, output, n_elements):
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, num_warps=config.num_warps, BLOCK_SIZE=config.kwargs['BLOCK_SIZE'])
    return kernel_fn

def vector_add_input_factory(size):
    """Input factory for autotuner"""
    def factory():
        x = torch.rand(size, device='cuda', dtype=torch.float32)
        y = torch.rand(size, device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)
        return (x, y, output, size)
    return factory
