import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata
from config import CONFIG
from hashi import HASHISampler
from metrics import evaluate_all
from tqdm import tqdm

# Experiment Config
SAMPLING_BUDGETS = [250, 500, 1000, 1500, 2000, 3000]
OUTPUT_DIR = "comparison_results"

class BaselineSampler:
    def __init__(self, wsi_path, gt_mask_shape):
        self.wsi_path = wsi_path
        self.map_w, self.map_h = gt_mask_shape[1], gt_mask_shape[0]
        self.helper = HASHISampler(wsi_path, None)
        self.tissue_mask = self.helper.tissue_mask
        self.valid_coords = self.helper.get_valid_tissue_coords() # list of (x, y)
        self.model = self.helper.model
        self.preprocess = self.helper.preprocess
        self.wsi_handler = self.helper.wsi_handler
        self.tile_size = self.helper.tile_size
        self.device = self.helper.device

    def run_inference(self, coords):
        return self.helper.run_inference(coords)

    def interpolate(self, coords, probs):
        if not coords:
            return np.zeros((self.map_h, self.map_w), dtype=np.float32)
            
        points = np.array(coords)
        values = np.array(probs)
        grid_x, grid_y = np.meshgrid(np.arange(self.map_w), np.arange(self.map_h))
        prob_map = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
        prob_map = np.nan_to_num(prob_map)
        prob_map[~self.tissue_mask] = 0
        return prob_map

    def sample_random(self, n_samples):
        indices = np.random.choice(len(self.valid_coords), min(n_samples, len(self.valid_coords)), replace=False)
        return [self.valid_coords[i] for i in indices]

    def sample_grid(self, n_samples):
        T = len(self.valid_coords)
        if n_samples >= T:
             return self.valid_coords
        stride = int(max(1, T // n_samples))
        return self.valid_coords[::stride]

def run_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wsi_path = CONFIG["wsi_path"]
    
    # Load GT
    gt_path = "Y_dense.npy"
    if not os.path.exists(gt_path):
        print("GT not found, cannot calculate metrics")
        return

    Y_dense = np.load(gt_path)
    # HASHI init to get shapes
    temp = HASHISampler(wsi_path, None)
    gt_mask_small = cv2.resize(Y_dense.astype(np.uint8), (temp.map_w, temp.map_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Initialize Baselines
    baseline_runner = BaselineSampler(wsi_path, gt_mask_small.shape)
    
    results = {
        "HASHI": {"Dice": [], "PSNR": [], "SSIM": [], "Samples": []},
        "Random": {"Dice": [], "PSNR": [], "SSIM": [], "Samples": []},
        "Grid":   {"Dice": [], "PSNR": [], "SSIM": [], "Samples": []}
    }
    
    print(f"Starting comparison on {len(SAMPLING_BUDGETS)} budget levels...")
    
    # 1. HASHI Run
    print("\n--- Running HASHI ---")
    hashi = HASHISampler(wsi_path, None)
    
    # Init 200
    current_n = hashi.initial_sampling(200)
    hashi.interpolate_map()
    
    max_budget = max(SAMPLING_BUDGETS)
    hashi_log = []
    
    # Initial
    m = evaluate_all(hashi.prob_map, gt_mask_small)
    m["Samples"] = current_n
    hashi_log.append(m)
    
    while current_n < max_budget + 500: # buffer
        hashi.compute_gradients()
        new = hashi.adaptive_sampling(100) # step 100
        if new == 0: break
        hashi.interpolate_map()
        current_n += new
        
        m = evaluate_all(hashi.prob_map, gt_mask_small)
        m["Samples"] = current_n
        hashi_log.append(m)
        print(f"HASHI: {current_n} samples, Dice: {m['Dice']:.4f}")

    # 2. Baseline Runs: Optimized
    
    # Define points to evaluate
    baseline_points = list(range(200, max_budget + 500, 200))
    
    print("\n--- Running Baselines (Optimized) ---")
    max_b = max(baseline_points)
    
    # -- Random --
    print(f"Random: Generating {max_b} samples...")
    # Sample max budget once
    all_random_coords = baseline_runner.sample_random(max_b)
    # Run inference on all
    all_random_probs = baseline_runner.run_inference(all_random_coords)
    
    random_log = []
    for n in tqdm(baseline_points, desc="Random Subsets"):
        coords = all_random_coords[:n]
        probs = all_random_probs[:n]
        pmap = baseline_runner.interpolate(coords, probs)
        m = evaluate_all(pmap, gt_mask_small)
        m["Samples"] = n
        random_log.append(m)

    # -- Grid --
    grid_log = []
    print("Grid: Running per-budget (geometry changes)...")
    
    memo = {}
    original_run_inference = baseline_runner.run_inference
    
    def get_probs(coords):
        missing = [c for c in coords if c not in memo]
        if missing:
            new_probs = original_run_inference(missing)
            for c, p in zip(missing, new_probs):
                memo[c] = p
        return [memo[c] for c in coords]
        
    baseline_runner.run_inference = get_probs
    
    for n in tqdm(baseline_points, desc="Grid budgets"):
        coords = baseline_runner.sample_grid(n)
        probs = baseline_runner.run_inference(coords)
        pmap = baseline_runner.interpolate(coords, probs)
        m = evaluate_all(pmap, gt_mask_small)
        m["Samples"] = n
        grid_log.append(m)
        
    # Save Results
    def save_log(log, name):
        df = pd.DataFrame(log)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_results.csv"), index=False)
        return df
        
    df_hashi = save_log(hashi_log, "HASHI")
    df_random = save_log(random_log, "Random")
    df_grid = save_log(grid_log, "Grid")
    
    # Plotting
    metrics = ["Dice", "PSNR", "SSIM"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(df_hashi["Samples"], df_hashi[metric], 'r-o', label="HASHI")
        plt.plot(df_random["Samples"], df_random[metric], 'b--x', label="Random")
        plt.plot(df_grid["Samples"], df_grid[metric], 'g-.s', label="Grid")
        
        plt.xlabel("Number of Samples")
        plt.ylabel(metric)
        plt.title(f"Sampling Strategy Comparison: {metric}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, f"compare_{metric}.png"))
        plt.close()
        
    print(f"\n[Comparison] Done. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_comparison()
