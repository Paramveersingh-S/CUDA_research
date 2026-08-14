import torch
from torch.utils.cpp_extension import load
import os

# JIT compile the CUDA kernel
_module_path = os.path.dirname(__file__)
vector_add_module = load(
    name="vector_add_cuda",
    sources=[os.path.join(_module_path, "vector_add.cu")],
    verbose=True
)

def add(x, y):
    return vector_add_module.add(x, y)
