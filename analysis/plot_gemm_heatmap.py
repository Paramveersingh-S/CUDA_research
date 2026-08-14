import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_heatmap(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['energy_per_call_j'])
    
    gemm_df = df[df['kernel'] == 'matmul'].copy()
    if gemm_df.empty:
        print("No GEMM data found in CSV.")
        return
        
    shapes = gemm_df['shape_str'].unique()
    
    for shape in shapes:
        subset = gemm_df[gemm_df['shape_str'] == shape].copy()
        
        block_m_list = []
        warps_list = []
        for cstr in subset['config_str']:
            # parse block size M and num_warps
            parts = cstr.split(',')
            bm = None
            w = None
            for p in parts:
                if 'BLOCK_SIZE_M' in p:
                    bm = int(p.split(':')[1].strip())
                elif 'num_warps' in p:
                    w = int(p.split(':')[1].strip())
            block_m_list.append(bm)
            warps_list.append(w)
            
        subset['BLOCK_SIZE_M'] = block_m_list
        subset['num_warps'] = warps_list
        
        # group by block size and warps and find minimum energy
        pivot = subset.groupby(['BLOCK_SIZE_M', 'num_warps'])['energy_per_call_j'].min().reset_index()
        pivot['energy_mj'] = pivot['energy_per_call_j'] * 1000
        
        heatmap_data = pivot.pivot(index='BLOCK_SIZE_M', columns='num_warps', values='energy_mj')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={'label': 'Energy (mJ)'})
        plt.title(f"GEMM Energy Heatmap\nShape: {shape}")
        plt.xlabel("Num Warps")
        plt.ylabel("Block Size M")
        
        safe_shape = str(shape).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
        out_path = os.path.join(output_dir, f"matmul_{safe_shape}_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/content/drive/MyDrive/greentune_results/full_sweep.csv")
    parser.add_argument("--out", type=str, default="analysis/plots")
    args = parser.parse_args()
    
    if os.path.exists(args.csv):
        generate_heatmap(args.csv, args.out)
