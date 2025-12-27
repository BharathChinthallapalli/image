import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from config import CONFIG

class WSIHandler:
    """
    Handler class for Whole Slide Image operations.
    Provides methods for loading, patch extraction, and tissue mask generation.
    """
    
    def __init__(self, wsi_path):
        """Initialize WSI handler."""
        self.slide = openslide.OpenSlide(wsi_path)
        self.w, self.h = self.slide.dimensions
        print(f"[WSIHandler] Loaded slide: {wsi_path}")
        print(f"[WSIHandler] Level-0 size: {self.w} x {self.h}")
        print(f"[WSIHandler] Levels available: {self.slide.level_count}")
        print(f"[WSIHandler] Level dimensions: {self.slide.level_dimensions}")

    def get_patch(self, x, y, size=256, level=0):
        """Extract an RGB patch from the slide."""
        return self.slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB")

    def get_thumbnail_auto(self, max_size=2048):
        """Get a downsampled thumbnail maintaining aspect ratio."""
        W0, H0 = self.w, self.h
        if max(W0, H0) <= max_size:
            target_size = (W0, H0)
        else:
            if W0 >= H0:
                target_size = (max_size, int(H0 * max_size / W0))
            else:
                target_size = (int(W0 * max_size / H0), max_size)
        return self.slide.get_thumbnail(target_size).convert("RGB")

    def generate_tissue_mask(self, max_thumb=2048, s_min=0.1, v_min=0.2, v_max=0.8):
        """Generate tissue mask using HSV thresholding with Otsu's method."""
        thumb = self.get_thumbnail_auto(max_thumb)
        W_t, H_t = thumb.size
        img_np = np.array(thumb).astype(np.float32) / 255.0

        # Convert to HSV
        hsv = rgb2hsv(img_np)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # Otsu thresholding
        thr_S = threshold_otsu(S)
        thr_V = threshold_otsu(V)

        # Combine Otsu with fixed constraints
        base_mask = (S > thr_S) & (V < thr_V)
        constraint_mask = (V >= v_min) & (V <= v_max) & (S > s_min)
        tissue_mask = base_mask & constraint_mask

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = tissue_mask.astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask = mask_u8.astype(bool)

        scale_x = self.w / float(W_t)
        scale_y = self.h / float(H_t)

        return mask, scale_x, scale_y

    def close(self):
        """Close the slide handle."""
        self.slide.close()
        print("[WSIHandler] Slide closed.")
    
if __name__ == "__main__":
    # Load WSI and display thumbnail
    try:
        wsi = WSIHandler(CONFIG["wsi_path"])

        thumb = wsi.get_thumbnail_auto(max_size=1024)
        plt.figure(figsize=(8, 8))
        plt.imshow(thumb)
        plt.title("WSI Thumbnail")
        plt.axis("off")
        print("Displaying thumbnail... (Close window to continue if running interactively)")
        # plt.show() # Commented out to prevent blocking in headless/agent environment, but user can uncomment
        plt.savefig("thumbnail_preview.png")
        print("Thumbnail saved to thumbnail_preview.png")
        
    except Exception as e:
        print(f"Error loading WSI: {e}")
