import torch
from torch.utils.cpp_extension import load
import os

_module_path = os.path.dirname(__file__)
rmsnorm_module = load(
    name="rmsnorm_cuda",
    sources=[os.path.join(_module_path, "rmsnorm.cu")],
    verbose=False
)

def rmsnorm(x, weight, eps=1e-5):
    return rmsnorm_module.rmsnorm(x, weight, eps)
