# --- environment bootstrap ---
import os
import subprocess
import json
import datetime

# Usually in Colab you run pip/apt via `!pip install ...`, but in a script we run them like this:
subprocess.run(["pip", "install", "-q", "triton", "pynvml", "optuna", "pandas", "matplotlib", "seaborn", "scipy", "statsmodels"])

import torch
import triton

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("gpu:", torch.cuda.get_device_name(0))

os.makedirs('/content/drive/MyDrive/greentune_results', exist_ok=True)  # after drive.mount()

# If running locally on Colab, we would import drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Colab environment or drive mount failed.")

# Attempt clock locking; record whether it actually worked
lock_result = subprocess.run(
    ["nvidia-smi", "-lgc", "300,1590"], capture_output=True, text=True
)
CLOCKS_LOCKED = (lock_result.returncode == 0)
print("Clock lock attempt:", "SUCCESS" if CLOCKS_LOCKED else "FAILED - report unlocked-clock results")

meta = dict(
    timestamp=str(datetime.datetime.utcnow()),
    gpu_name=torch.cuda.get_device_name(0),
    torch_version=torch.__version__,
    cuda_version=torch.version.cuda,
    triton_version=triton.__version__,
    clocks_locked=CLOCKS_LOCKED,
)
json.dump(meta, open('/content/drive/MyDrive/greentune_results/session_meta.json', 'a'))
