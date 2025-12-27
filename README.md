# HASHI: Hierarchical Adaptive Sampling for High-resolution Imaging

This repository contains the implementation of the HASHI algorithm for efficient Whole Slide Image (WSI) analysis.

## Project Structure

* `hashi.py`: Core algorithm implementation (`HASHISampler` class).
* `run_hashi.py`: Script to run a single experiment on a slide.
* `compare_samplers.py`: Script to compare HASHI vs Random/Grid sampling.
* `metrics.py`: Evaluation metrics (Dice, IoU, PSNR, SSIM).
* `WSI_load.py`: Utility to handle WSI loading (requires `openslide`).
* `config.py`: Configuration parameters.
* `model_utils.py`: Helper functions for model loading (ResNet50).
* `requirements.txt`: Python dependencies.

## Usage on Google Colab (Recommended)

**Step 1**: Upload this entire folder to your Google Drive or Colab environment.

**Step 2**: Open `HASHI_Workflow.ipynb`. This notebook provides a complete, interactive pipeline that will:

1. Install all dependencies (`openslide-tools`, etc.).
2. Allow you to input your WSI path.
3. Run the entire HASHI analysis and visualize the results.

## Manual Usage (Script)

If you prefer running scripts directly:

1. **Upload Data**:

   ```bash
   !pip install -r requirements.txt
   !apt-get install openslide-tools
   ```

2. **Run HASHI**:

   ```python
   !python run_hashi.py --wsi_path "/content/drive/MyDrive/slide.tif" --xml_path "/content/drive/MyDrive/annotation.xml" --output_dir "/content/output"
   ```

3. **Compare Sampling Strategies**:

   ```python
   !python compare_samplers.py
   ```

## Requirements

* Python 3.8+
* PyTorch
* OpenSlide
* NumPy, Pandas, Matplotlib, Scikit-Image
