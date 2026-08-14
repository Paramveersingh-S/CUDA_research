import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_comparison(grid_csv, optuna_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load grid
    df_grid = pd.read_csv(grid_csv)
    df_grid = df_grid[(df_grid['kernel'] == 'matmul') & (df_grid['shape_str'] == '(4096, 4096, 4096)')]
    df_grid = df_grid.dropna(subset=['energy_per_call_j'])
    
    # Load optuna
    df_opt = pd.read_csv(optuna_csv)
    df_opt = df_opt.dropna(subset=['energy_per_call_j'])
    
    # Calculate True Pareto Front from grid
    pts = df_grid[['median_latency_s', 'energy_per_call_j']].values
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, c in enumerate(pts):
        if is_pareto[i]:
            dominated = np.all(pts >= c, axis=1) & np.any(pts > c, axis=1)
            is_pareto[dominated] = False
            
    df_grid['is_pareto'] = is_pareto
    pareto_front = df_grid[df_grid['is_pareto']].sort_values(by='median_latency_s')
    
    plt.figure(figsize=(10, 6))
    
    # Plot true grid background
    plt.scatter(
        df_grid['median_latency_s'] * 1000, 
        df_grid['energy_per_call_j'] * 1000, 
        c='lightgray', alpha=0.5, label='Exhaustive Grid (All Points)'
    )
    
    plt.plot(
        pareto_front['median_latency_s'] * 1000, 
        pareto_front['energy_per_call_j'] * 1000, 
        c='red', linestyle='--', linewidth=1.5, label='True Pareto Front (Grid)'
    )
    
    # Plot Optuna points
    plt.scatter(
        df_opt['median_latency_s'] * 1000, 
        df_opt['energy_per_call_j'] * 1000, 
        c='blue', marker='x', s=100, linewidths=2, label='Optuna Explored Points'
    )
    
    # Extract Optuna front
    opts = df_opt[['median_latency_s', 'energy_per_call_j']].values
    if len(opts) > 0:
        is_opt_pareto = np.ones(opts.shape[0], dtype=bool)
        for i, c in enumerate(opts):
            if is_opt_pareto[i]:
                dominated = np.all(opts >= c, axis=1) & np.any(opts > c, axis=1)
                is_opt_pareto[dominated] = False
                
        df_opt['is_pareto'] = is_opt_pareto
        opt_front = df_opt[df_opt['is_pareto']].sort_values(by='median_latency_s')
        
        plt.plot(
            opt_front['median_latency_s'] * 1000, 
            opt_front['energy_per_call_j'] * 1000, 
            c='darkblue', marker='o', linewidth=2, label='Optuna Found Front'
        )
    
    plt.title('Cost-Model-Guided Search (Optuna) vs Exhaustive Grid\nKernel: GEMM (4096x4096x4096)')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Energy (mJ)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_path = os.path.join(output_dir, "optuna_vs_grid_pareto.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="/content/drive/MyDrive/greentune_results/full_sweep.csv")
    parser.add_argument("--optuna", type=str, default="/content/drive/MyDrive/greentune_results/optuna_sweep.csv")
    parser.add_argument("--out", type=str, default="analysis/plots")
    args = parser.parse_args()
    
    if os.path.exists(args.grid) and os.path.exists(args.optuna):
        plot_comparison(args.grid, args.optuna, args.out)
    else:
        print("Missing CSVs. Ensure both full_sweep.csv and optuna_sweep.csv exist.")
