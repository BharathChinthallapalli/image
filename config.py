import os

# Configuration
CONFIG = {
    'wsi_path': '/content/wsi/tumor_078.tif', # Placeholder for Colab
    'xml_path': '/content/wsi/tumor_078.xml',
    'classifier_path': 'resnet50_weights.pth',
    'output_path': '/content/output/tumor_078_heatmap.png',
    'patch_output_dir': '/content/output/tiles',
    'csv_output_path': '/content/output/tiles_meta.csv',
    'level': 0,
    'patch_size': 256,
    'stride': 256,
    'sampling_budget': 5000,
    'vis_downsample': 32,
    'tissue_frac_threshold': 0.4,
    'downsample_for_otsu': 32,
}

# Create output directories
# os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True) # Usually handled by run scripts
