import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from WSI_load import WSIHandler

def generate_heatmap(csv_path, wsi_path, output_path, downsample_factor=32):
    """
    Generate a probability heatmap from WSInfer CSV results.
    """
    print(f"[Heatmap] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load WSI for dimensions and thumbnail
    wsi = WSIHandler(wsi_path)
    W0, H0 = wsi.w, wsi.h
    
    # Create thumbnail
    # We want a specific downsample factor to map coordinates easily
    # Target size
    Wt = W0 // downsample_factor
    Ht = H0 // downsample_factor
    
    print(f"[Heatmap] Generating thumbnail (downsample={downsample_factor})...")
    thumb = wsi.slide.get_thumbnail((Wt, Ht)).convert("RGB")
    thumb_np = np.array(thumb)
    
    # Resize thumbnail to exact target dimensions if get_thumbnail was approximate
    if thumb_np.shape[:2] != (Ht, Wt):
        thumb_np = cv2.resize(thumb_np, (Wt, Ht))
    
    # Create heatmap array
    heatmap = np.zeros((Ht, Wt), dtype=np.float32)
    
    print("[Heatmap] Mapping probabilities...")
    # Iterate through CSV
    # Columns: minx, miny, width, height, prob_Metastasis or probability
    
    # Check column names
    prob_col = 'prob_Metastasis' if 'prob_Metastasis' in df.columns else 'probability'
    if prob_col not in df.columns:
         # Fallback for standard wsinfer output might be different?
         # wsinfer usually outputs 'prob_<classname>'
         cols = [c for c in df.columns if c.startswith('prob_')]
         if cols:
             prob_col = cols[-1] # Pick the last class (usually Tumor/Positive)
         else:
             print("Warning: No probability column found.")
             return

    for idx, row in df.iterrows():
        x = int(row['minx'])
        y = int(row['miny'])
        w = int(row['width'])
        h = int(row['height'])
        prob = float(row[prob_col])
        
        # Scale to thumbnail
        x_t = x // downsample_factor
        y_t = y // downsample_factor
        w_t = max(1, w // downsample_factor)
        h_t = max(1, h // downsample_factor)
        
        heatmap[y_t:y_t+h_t, x_t:x_t+w_t] = prob

    # Apply colormap using Matplotlib for better visualization with colorbar
    print(f"[Heatmap] Saving plot with colorbar to {output_path}")
    
    plt.figure(figsize=(12, 10))
    plt.imshow(thumb_np)
    plt.imshow(heatmap, cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar()
    cbar.set_label('Tumor Probability', rotation=270, labelpad=15)
    plt.title(f"Tumor Probability Heatmap\n(Dense Inference)")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save raw heatmap
    heatmap_path = output_path.replace(".png", "_raw_prob.png")
    plt.imsave(heatmap_path, heatmap, cmap='gray', vmin=0.0, vmax=1.0)
    
    wsi.close()
    print("[Heatmap] Done.")
