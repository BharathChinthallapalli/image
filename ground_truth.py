import numpy as np
import cv2
import os
from config import CONFIG
from WSI_load import WSIHandler
from annotation_parser import AnnotationParser
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

def compute_tissue_and_dataset(wsi_handler, annotation_parser, map_w, map_h):
    """
    Computes tissue mask and Ground Truth mask at the resolution of the probability map (map_w, map_h).
    """
    W0, H0 = wsi_handler.w, wsi_handler.h
    
    # Scale Factors
    scale_x = W0 / map_w
    scale_y = H0 / map_h
    
    print(f"[GT] Generating masks for map size: {map_w}x{map_h} (Downsample ~{scale_x:.2f})")
    
    # 1. Tissue Mask (Otsu from thumbnail)
    # We use generate_tissue_mask from handler but need to resize to map_w, map_h
    # Handlers generate_tissue_mask returns logic, let's reuse it carefully
    # To get best quality, we get a thumbnail close to target size
    
    thumb_size = max(map_w, map_h)
    mask_raw, _, _ = wsi_handler.generate_tissue_mask(max_thumb=thumb_size)
    
    # Resize to exact map dimensions
    tissue_mask = cv2.resize(mask_raw.astype(np.uint8), (map_w, map_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # 2. Annotation Mask
    # If no parser provided, return None
    if annotation_parser is None:
        return tissue_mask, None
        
    ann_mask = annotation_parser.create_mask(
        shape=(map_h, map_w),
        downsample_factor=scale_x, # Assuming square pixels mostly
        tumor_only=True
    ).astype(bool)
    
    return tissue_mask, ann_mask

if __name__ == "__main__":
    # Example usage suited for Colab
    # We assume 'hashi' module usage logic for map size 
    # (W // 256, H // 256)
    
    wsi_path = float(CONFIG["wsi_path"]) if "wsi_path" in CONFIG else None # logic placeholder
    pass
