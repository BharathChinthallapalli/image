import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
from WSI_load import WSIHandler

def extract_tiles_direct(wsi_path, output_dir, patch_size=256, stride=256):
    """
    Extract tissue tiles based on simple tissue detection and save them.
    Returns list of saved tile metadata.
    """
    print(f"[ExtractTiles] processing {wsi_path}...")
    wsi = WSIHandler(wsi_path)
    
    # Tissue Mask (Level-0 coords)
    mask_thumb, sx, sy = wsi.generate_tissue_mask(max_thumb=2048)
    # Resize mask to Level 0 resolution approx? No, map coords.
    # Let's just iterate logical grid and check mask
    
    W, H = wsi.w, wsi.h
    
    # Mask is small, we map coords to it
    H_m, W_m = mask_thumb.shape
    
    rows = range(0, H, stride)
    cols = range(0, W, stride)
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "tiles.csv")
    
    extracted = []
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["minx", "miny", "width", "height", "file_path"])
        
        count = 0
        for y in tqdm(rows, desc="Extracting Tiles"):
            for x in cols:
                # Check tissue intersection
                # Map x,y to mask coords
                xm = int(x / sx)
                ym = int(y / sy)
                
                if 0 <= ym < H_m and 0 <= xm < W_m:
                    if mask_thumb[ym, xm]: # Simple point check or region check
                        # Extract
                        fname = f"tile_{x}_{y}.jpg"
                        fpath = os.path.join(output_dir, fname)
                        
                        # In a real heavy run we might skip saving provided we run inference in-memory
                        # But for "Standard Pipeline" we save.
                        patch = wsi.get_patch(x, y, size=patch_size)
                        patch.save(fpath)
                        
                        writer.writerow([x, y, patch_size, patch_size, fpath])
                        extracted.append((x, y))
                        count += 1
                        
    print(f"[ExtractTiles] Extracted {count} tiles to {output_dir}")
    wsi.close()
    return csv_path
