import pandas as pd, os

RESULTS_PATH = "/content/drive/MyDrive/greentune_results/full_sweep.csv"

def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(RESULTS_PATH)
    df.to_csv(RESULTS_PATH, mode="a", header=header, index=False)

def already_done(kernel, config_key, shape):
    if not os.path.exists(RESULTS_PATH):
        return False
    df = pd.read_csv(RESULTS_PATH)
    return ((df.kernel == kernel) & (df.config_key == config_key) & (df.shape == shape)).any()

# In the sweep loop: if already_done(...): continue before every benchmark call.
