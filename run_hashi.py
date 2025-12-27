import os
import argparse
from config import CONFIG
from hashi import HASHISampler

import numpy as np
import cv2
import matplotlib.pyplot as plt

def run_hashi_experiment(wsi_path, tissue_mask_path, output_dir, n_iterations=20, initial_samples=200, adaptive_samples=50):
    print(f"[RunHASHI] Starting HASHI experiment...")
    print(f"[RunHASHI] WSI: {wsi_path}")
    print(f"[RunHASHI] Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Sampler
    sampler = HASHISampler(wsi_path, tissue_mask_path)
    
    # Load Ground Truth if available
    gt_path = "Y_dense.npy"
    gt_mask_small = None
    if os.path.exists(gt_path):
        print(f"[RunHASHI] Loading Ground Truth from {gt_path}...")
        Y_dense = np.load(gt_path)
        # Resize to match sampler map size
        gt_mask_small = cv2.resize(Y_dense.astype(np.uint8), (sampler.map_w, sampler.map_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        print(f"[RunHASHI] GT resized to {gt_mask_small.shape}")
    else:
        print("[RunHASHI] Warning: Y_dense.npy not found. Dice scores will not be computed.")

    dice_scores = []
    
    # Step 1: Initial Sampling (Exploration)
    print("\n--- Iteration 0: Initial Sampling ---")
    sampler.initial_sampling(n_samples=initial_samples)
    sampler.interpolate_map()
    
    if gt_mask_small is not None:
        dice = sampler.compute_dice(gt_mask_small)
        dice_scores.append(dice)
        print(f"Iteration 0 Dice: {dice:.4f}")
        
    sampler.save_state(output_dir, iteration=0)
    
    # Step 2: Adaptive Loop
    for i in range(1, n_iterations + 1):
        print(f"\n--- Iteration {i}: Adaptive Sampling ---")
        
        # Compute Gradients
        sampler.compute_gradients()
        
        # Sample based on gradients
        n_new = sampler.adaptive_sampling(n_samples=adaptive_samples)
        
        if n_new == 0:
            print("[RunHASHI] Stopping early (no more candidates).")
            break
            
        # Update Map
        sampler.interpolate_map()
        
        # Compute Metrics
        if gt_mask_small is not None:
            dice = sampler.compute_dice(gt_mask_small)
            dice_scores.append(dice)
            print(f"Iteration {i} Dice: {dice:.4f}")
        
        # Save Visualization
        sampler.save_state(output_dir, iteration=i)
        
    # Plot Metrics
    if dice_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(dice_scores)), dice_scores, marker='o')
        plt.title("HASHI Performance (Dice Score vs Iterations)")
        plt.xlabel("Iteration")
        plt.ylabel("Dice Score")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "hashi_dice_curve.png"))
        print(f"[RunHASHI] Saved Dice curve to {output_dir}/hashi_dice_curve.png")
        
    print(f"\n[RunHASHI] Experiment completed. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_path", type=str, default=CONFIG["wsi_path"])
    parser.add_argument("--xml_path", type=str, default=CONFIG["xml_path"])
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(CONFIG["output_path"]), "hashi_results"))
    args = parser.parse_args()
    
    # Run with on-the-fly mask generation (tissue_mask_path=None)
    run_hashi_experiment(args.wsi_path, None, args.output_dir, n_iterations=30, initial_samples=200, adaptive_samples=100)
