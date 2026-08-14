import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

def identify_pareto(df):
    """
    Given a dataframe with median_latency_s and energy_per_call_j,
    returns a boolean mask of the Pareto-optimal points (minimize both).
    """
    pts = df[['median_latency_s', 'energy_per_call_j']].values
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, c in enumerate(pts):
        if is_pareto[i]:
            # A point c dominates another point if it is <= in all dimensions and < in at least one
            # The mask keeps points that are NOT strictly dominated by c
            dominated = np.all(pts >= c, axis=1) & np.any(pts > c, axis=1)
            is_pareto[dominated] = False
    return is_pareto

def generate_plots(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Drop rows where energy is NaN
    df = df.dropna(subset=['energy_per_call_j'])
    
    kernels = df['kernel'].unique()
    
    for kernel in kernels:
        kernel_df = df[df['kernel'] == kernel]
        shapes = kernel_df['shape_str'].unique()
        
        for shape in shapes:
            subset = kernel_df[kernel_df['shape_str'] == shape].copy()
            if subset.empty:
                continue
                
            # Compute Pareto
            is_pareto = identify_pareto(subset)
            subset['is_pareto'] = is_pareto
            
            pareto_front = subset[subset['is_pareto']].sort_values(by='median_latency_s')
            non_pareto = subset[~subset['is_pareto']]
            
            plt.figure(figsize=(10, 6))
            
            # Plot non-pareto points
            plt.scatter(
                non_pareto['median_latency_s'] * 1000, 
                non_pareto['energy_per_call_j'] * 1000, 
                c='blue', alpha=0.5, label='Suboptimal'
            )
            
            # Plot pareto front
            plt.plot(
                pareto_front['median_latency_s'] * 1000, 
                pareto_front['energy_per_call_j'] * 1000, 
                c='red', marker='o', linewidth=2, markersize=8, label='Pareto Front'
            )
            
            # Add knee point identification (heuristic: point closest to origin after normalization)
            if len(pareto_front) > 0:
                norm_lat = (pareto_front['median_latency_s'] - pareto_front['median_latency_s'].min()) / \
                           (pareto_front['median_latency_s'].max() - pareto_front['median_latency_s'].min() + 1e-9)
                norm_nrg = (pareto_front['energy_per_call_j'] - pareto_front['energy_per_call_j'].min()) / \
                           (pareto_front['energy_per_call_j'].max() - pareto_front['energy_per_call_j'].min() + 1e-9)
                
                dist_to_origin = np.sqrt(norm_lat**2 + norm_nrg**2)
                knee_idx = dist_to_origin.argmin()
                knee_pt = pareto_front.iloc[knee_idx]
                
                plt.scatter(
                    knee_pt['median_latency_s'] * 1000, 
                    knee_pt['energy_per_call_j'] * 1000, 
                    c='gold', s=200, edgecolors='black', zorder=5, label='Optimal "Knee" Point'
                )

            plt.title(f'Pareto Front: {kernel} (Shape: {shape})')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Energy (mJ)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Clean up filename
            safe_shape = str(shape).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
            out_path = os.path.join(output_dir, f"{kernel}_{safe_shape}_pareto.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/content/drive/MyDrive/greentune_results/full_sweep.csv")
    parser.add_argument("--out", type=str, default="analysis/plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: Could not find {args.csv}")
        sys.exit(1)
        
    generate_plots(args.csv, args.out)
