import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from huggingface_hub import hf_hub_download
import json
import os

model_path = "ibm-granite/granite-vision-3.2-2b"

# Check the model config first
print("=== MODEL CONFIG INSPECTION ===")
config = AutoConfig.from_pretrained(model_path)

# Print all config attributes
print(f"Model type: {config.model_type}")
print(f"Torch dtype: {getattr(config, 'torch_dtype', 'Not specified')}")

# Check for quantization config
if hasattr(config, 'quantization_config'):
    print(f"Has quantization_config: True")
    quant_config = config.quantization_config
    print(f"Quantization config: {quant_config}")
    
    # Print specific quantization settings
    if hasattr(quant_config, 'quant_method'):
        print(f"Quantization method: {quant_config.quant_method}")
    if hasattr(quant_config, 'bits'):
        print(f"Bits: {quant_config.bits}")
    if hasattr(quant_config, 'load_in_8bit'):
        print(f"Load in 8bit: {quant_config.load_in_8bit}")
    if hasattr(quant_config, 'load_in_4bit'):
        print(f"Load in 4bit: {quant_config.load_in_4bit}")
else:
    print("Has quantization_config: False")

# Check all config attributes that might be related to quantization
print("\n=== ALL CONFIG ATTRIBUTES ===")
for attr in dir(config):
    if not attr.startswith('_') and 'quant' in attr.lower():
        value = getattr(config, attr)
        print(f"{attr}: {value}")

print("\n=== MODEL FILES INSPECTION ===")
try:
    # Try to download and inspect config.json directly
    config_path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(config_path, 'r') as f:
        raw_config = json.load(f)
    
    print("Raw config.json quantization-related fields:")
    for key, value in raw_config.items():
        if 'quant' in key.lower() or 'dtype' in key.lower():
            print(f"  {key}: {value}")
            
    # Check if there are any GGUF or quantized model files
    from huggingface_hub import list_repo_files
    files = list_repo_files(model_path)
    
    print(f"\nModel files:")
    for file in files:
        if any(ext in file.lower() for ext in ['gguf', 'q4', 'q8', 'int4', 'int8']):
            print(f"  Quantized file found: {file}")
        elif file.endswith('.safetensors') or file.endswith('.bin'):
            print(f"  Weight file: {file}")
            
except Exception as e:
    print(f"Error inspecting files: {e}")

print("\n=== ATTEMPTING MODEL LOAD WITH DEFAULT SETTINGS ===")
try:
    # Load processor first
    processor = AutoProcessor.from_pretrained(model_path)
    print("✓ Processor loaded successfully")
    
    # Try loading model with minimal settings
    print("Attempting model load with default settings...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )
    print("✓ Model loaded successfully with default settings")
    
    # Check actual model properties
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Check memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU memory used: {allocated:.2f}GB")
    
    # Check if model has quantization
    if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
        print(f"Model quantization config: {model.config.quantization_config}")
    
    # Check model structure
    print("\n=== MODEL STRUCTURE ===")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight_dtype = module.weight.dtype
            if 'quant' in name.lower() or weight_dtype not in [torch.float16, torch.float32]:
                print(f"{name}: {weight_dtype}")
                break  # Just show first few to avoid spam
        if len(name.split('.')) <= 2:  # Only top-level modules
            print(f"{name}: {type(module).__name__}")

except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Error type: {type(e).__name__}")

print("\n=== ATTEMPTING LOAD WITH DIFFERENT DTYPE ===")
try:
    # Try with float32 to avoid Half/Char dtype mismatch
    model_fp32 = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("✓ Model loaded successfully with float32")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU memory used (float32): {allocated:.2f}GB")
        
except Exception as e:
    print(f"Error loading with float32: {e}")
