import torch
from torch.utils.cpp_extension import load
import os

_module_path = os.path.dirname(__file__)
softmax_module = load(
    name="softmax_cuda",
    sources=[os.path.join(_module_path, "softmax.cu")],
    verbose=False
)

def softmax(x):
    return softmax_module.softmax(x)
