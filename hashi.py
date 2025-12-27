import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import sobel
from torchvision import transforms
from tqdm import tqdm
from WSI_load import WSIHandler
from model_utils import load_model_with_jit_weights

class HASHISampler:
    def __init__(self, wsi_path, tissue_mask_path=None, device="cpu"):
        self.wsi_path = wsi_path
        self.device = device
        
        # Load WSI Handler
        self.wsi_handler = WSIHandler(wsi_path)
        self.W, self.H = self.wsi_handler.w, self.wsi_handler.h
        
        # Determine working resolution
        self.tile_size = 256
        self.map_w = self.W // self.tile_size
        self.map_h = self.H // self.tile_size
        print(f"[HASHI] Working map size: {self.map_w} x {self.map_h} (1 px = {self.tile_size}x{self.tile_size} L0)")

        # Load or Generate Tissue Mask
        if tissue_mask_path and os.path.exists(tissue_mask_path):
            print(f"[HASHI] Loading tissue mask from {tissue_mask_path}...")
            self.tissue_mask_orig = cv2.imread(tissue_mask_path, cv2.IMREAD_GRAYSCALE)
            if self.tissue_mask_orig is None:
                raise ValueError(f"Could not load tissue mask from {tissue_mask_path}")
            # Resize to map size
            self.tissue_mask = cv2.resize(self.tissue_mask_orig, (self.map_w, self.map_h), interpolation=cv2.INTER_NEAREST)
            self.tissue_mask = (self.tissue_mask > 0).astype(bool)
        else:
            print("[HASHI] Generating tissue mask on-the-fly...")
            # Generate mask using WSIHandler
            # We want a mask that roughly matches our map size
            # map_w is roughly W / 256. 
            # generate_tissue_mask takes max_thumb. 
            # If we set max_thumb to max(self.map_w, self.map_h), we get a mask close to desired resolution.
            max_dim = max(self.map_w, self.map_h)
            mask, sx, sy = self.wsi_handler.generate_tissue_mask(max_thumb=max_dim)
            
            # Resize exactly to map size
            self.tissue_mask = cv2.resize(mask.astype(np.uint8), (self.map_w, self.map_h), interpolation=cv2.INTER_NEAREST).astype(bool)

        
        # Load JIT Model directly (WSInfer style)
        print(f"[HASHI] Loading JIT model from wsinfer_zoo...")
        from wsinfer_zoo.client import load_torchscript_model_from_hf
        wrapper = load_torchscript_model_from_hf("kaczmarj/lymphnodes-tiatoolbox-resnet50.patchcamelyon")
        self.model = torch.jit.load(wrapper.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing from wsinfer config: Resize(96) -> ToTensor
        self.preprocess = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
        
        # State
        self.sampled_coords = [] # List of (x_grid, y_grid)
        self.sampled_probs = []  # List of float probabilities
        self.prob_map = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self.gradient_map = np.zeros((self.map_h, self.map_w), dtype=np.float32)

    def get_valid_tissue_coords(self):
        """Returns all (x, y) grid coordinates that are within the tissue mask."""
        ys, xs = np.where(self.tissue_mask)
        return list(zip(xs, ys))

    def run_inference(self, grid_coords):
        """
        Runs inference on the specified grid coordinates.
        grid_coords: list of (x_grid, y_grid)
        """
        new_probs = []
        batch_size = 32
        
        # Convert grid coords to L0 coords
        l0_coords = [(x * self.tile_size, y * self.tile_size) for x, y in grid_coords]
        
        batches = [l0_coords[i:i + batch_size] for i in range(0, len(l0_coords), batch_size)]
        
        for batch in tqdm(batches, desc="Running Inference"):
            tensors = []
            for x, y in batch:
                patch = self.wsi_handler.get_patch(x, y, size=self.tile_size)
                tensors.append(self.preprocess(patch))
            
            if not tensors:
                continue
                
            input_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy() # Class 1 = Tumor
                new_probs.extend(probs)
                
        return new_probs

    def initial_sampling(self, n_samples=100):
        """
        Randomly samples n_samples from the tissue region.
        """
        print(f"[HASHI] Initial sampling: {n_samples} tiles...")
        valid_coords = self.get_valid_tissue_coords()
        
        if len(valid_coords) < n_samples:
            selected_indices = np.arange(len(valid_coords))
        else:
            selected_indices = np.random.choice(len(valid_coords), n_samples, replace=False)
            
        selected_coords = [valid_coords[i] for i in selected_indices]
        
        # Run inference
        probs = self.run_inference(selected_coords)
        
        # Update state
        self.sampled_coords.extend(selected_coords)
        self.sampled_probs.extend(probs)
        
        return len(selected_coords)

    def interpolate_map(self):
        """
        Interpolates sparse samples to create a continuous probability map.
        """
        if not self.sampled_coords:
            return
            
        points = np.array(self.sampled_coords)
        values = np.array(self.sampled_probs)
        
        # Grid for interpolation
        grid_x, grid_y = np.meshgrid(np.arange(self.map_w), np.arange(self.map_h))
        
        # Interpolate (linear or cubic)
        # fill_value=0 for background
        self.prob_map = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
        
        # Mask out non-tissue regions
        self.prob_map[~self.tissue_mask] = 0
        
        # Handle NaNs from linear interpolation outside convex hull if any (fill_value handles most)
        self.prob_map = np.nan_to_num(self.prob_map)

    def compute_gradients(self):
        """
        Computes the gradient magnitude of the probability map.
        """
        # Sobel gradients
        sx = sobel(self.prob_map, axis=1, mode='constant')
        sy = sobel(self.prob_map, axis=0, mode='constant')
        self.gradient_map = np.hypot(sx, sy)
        
        # Mask out non-tissue
        self.gradient_map[~self.tissue_mask] = 0

    def compute_dice(self, gt_mask, threshold=0.5):
        """
        Computes Dice score against a ground truth mask (must be same shape as prob_map).
        """
        pred_mask = (self.prob_map > threshold)
        
        intersection = (pred_mask & gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
            
        return 2.0 * intersection / union

    def adaptive_sampling(self, n_samples=50):
        """
        Samples new tiles based on gradient magnitude.
        """
        print(f"[HASHI] Adaptive sampling: {n_samples} tiles...")
        
        # Get candidates (all tissue pixels not yet sampled)
        # For efficiency, we can just sample from the gradient distribution
        # or pick top-k gradients.
        # HASHI paper suggests QMC proportional to gradient.
        # Simplified: Weighted random sampling based on gradient magnitude.
        
        valid_coords = self.get_valid_tissue_coords()
        sampled_set = set(self.sampled_coords)
        candidates = [c for c in valid_coords if c not in sampled_set]
        
        if not candidates:
            print("[HASHI] No more candidates to sample.")
            return 0
            
        # Get gradients for candidates
        cand_grads = np.array([self.gradient_map[y, x] for x, y in candidates])
        
        # Add small epsilon to allow exploration of zero-gradient regions (exploration vs exploitation)
        weights = cand_grads + 1e-6
        weights /= weights.sum()
        
        # Sample
        if len(candidates) < n_samples:
            selected_indices = np.arange(len(candidates))
        else:
            selected_indices = np.random.choice(len(candidates), n_samples, replace=False, p=weights)
            
        selected_coords = [candidates[i] for i in selected_indices]
        
        # Run inference
        probs = self.run_inference(selected_coords)
        
        # Update state
        self.sampled_coords.extend(selected_coords)
        self.sampled_probs.extend(probs)
        
        return len(selected_coords)

    def save_state(self, output_dir, iteration):
        """
        Saves the current probability map and sampling points visualization.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 10))
        # plt.imshow(self.prob_map, cmap='jet', vmin=0, vmax=1) # Commented to avoid display issues in headless
        plt.imshow(self.prob_map, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(label='Tumor Probability')
        
        # Overlay points
        pts = np.array(self.sampled_coords)
        plt.scatter(pts[:, 0], pts[:, 1], c='white', s=2, alpha=0.5, label='Samples')
        
        plt.title(f"HASHI Iteration {iteration} ({len(pts)} samples)")
        plt.legend()
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"iter_{iteration:02d}_map.png"))
        plt.close()
        
        # Save Gradient Map
        plt.figure(figsize=(12, 10))
        plt.imshow(self.gradient_map, cmap='magma')
        plt.colorbar(label='Gradient Magnitude')
        plt.title(f"HASHI Iteration {iteration} Gradients")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"iter_{iteration:02d}_grads.png"))
        plt.close()

    def get_sparse_maps(self):
        """
        Returns the current sparse representation maps.
        Returns:
            H_sparse: 2D array with prob at sampled locs, 0 elsewhere.
            M: 2D binary mask (1 at sampled locs).
            U: 2D uncertainty map (gradients).
        """
        H_sparse = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        M = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
        
        for (x, y), p in zip(self.sampled_coords, self.sampled_probs):
            if 0 <= y < self.map_h and 0 <= x < self.map_w:
                H_sparse[y, x] = p
                M[y, x] = 1
                
        return H_sparse, M, self.gradient_map.copy()
