import pandas as pd
import os
import argparse

def generate_tables(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['energy_per_call_j'])
    
    out_table_lines = ["# Cross-Shape Optimal Configuration Stability\n"]
    
    kernels = df['kernel'].unique()
    for kernel in kernels:
        kernel_df = df[df['kernel'] == kernel]
        shapes = kernel_df['shape_str'].unique()
        
        out_table_lines.append(f"## Kernel: {kernel}\n")
        
        for shape in shapes:
            subset = kernel_df[kernel_df['shape_str'] == shape]
            if subset.empty: continue
            
            # Find minimum energy config
            best_nrg_idx = subset['energy_per_call_j'].idxmin()
            best_nrg_cfg = subset.loc[best_nrg_idx]
            
            # Find minimum latency config
            best_lat_idx = subset['median_latency_s'].idxmin()
            best_lat_cfg = subset.loc[best_lat_idx]
            
            out_table_lines.append(f"- **Shape {shape}**")
            out_table_lines.append(f"  - Min Energy: {best_nrg_cfg['config_str']} -> {best_nrg_cfg['energy_per_call_j']*1000:.2f} mJ")
            out_table_lines.append(f"  - Min Latency: {best_lat_cfg['config_str']} -> {best_lat_cfg['median_latency_s']*1000:.2f} ms\n")
            
    with open(os.path.join(output_dir, "stability_tables.md"), "w") as f:
        f.write("\n".join(out_table_lines))
    print(f"Saved: {os.path.join(output_dir, 'stability_tables.md')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/content/drive/MyDrive/greentune_results/full_sweep.csv")
    parser.add_argument("--out", type=str, default="analysis")
    args = parser.parse_args()
    
    if os.path.exists(args.csv):
        generate_tables(args.csv, args.out)
