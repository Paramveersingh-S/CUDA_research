import torch
from torch.utils.cpp_extension import load
import os

_module_path = os.path.dirname(__file__)
matmul_module = load(
    name="matmul_cuda",
    sources=[os.path.join(_module_path, "matmul.cu")],
    verbose=False
)

def matmul(a, b):
    return matmul_module.matmul(a, b)
