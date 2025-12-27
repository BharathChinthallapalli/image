import torch
import torchvision.models as models
from wsinfer_zoo.client import load_torchscript_model_from_hf

def load_model_with_jit_weights(model_name="kaczmarj/lymphnodes-tiatoolbox-resnet50.patchcamelyon"):
    """
    Loads a standard ResNet50 model with weights transferred from the wsinfer JIT model.
    """
    print(f"[ModelUtils] Loading JIT model weights from {model_name}...")
    try:
        # 1. Get JIT model
        wrapper = load_torchscript_model_from_hf(model_name)
        jit_model = torch.jit.load(wrapper.model_path)
        jit_state = jit_model.state_dict()
        
        # 2. Get Standard ResNet50
        model = models.resnet50()
        
        # Modify FC layer to match JIT model (2 classes)
        model.fc = torch.nn.Linear(2048, 2)
        
        # 3. Transfer weights
        print("[ModelUtils] Transferring weights to standard ResNet50...")
        missing, unexpected = model.load_state_dict(jit_state, strict=True)
        print(f"[ModelUtils] Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        return model
    except Exception as e:
        print(f"[ModelUtils] Error loading weights: {e}")
        raise e
