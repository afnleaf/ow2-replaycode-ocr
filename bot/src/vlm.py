# external
import cv2 as cv
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch
import re
import gc
import os
import time
import warnings
# local
import ocr

warnings.filterwarnings("ignore", message="The default value of the antialias parameter")

# to enable testing functionality
load_dotenv()
TEST: bool = True if str(os.getenv("ENVIRONMENT")) == "test" else False
print("Bool Testing:", TEST)
# the directory we will store our images
output_append = "/app/output/"

#prompt = "What is the 6 character code in this image? The output must contain only 6 characters. These are the valid characters: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" 
prompt = "What is the exactly 6 character code in this image? The code must be exsactly 6 characters. Answer with the code only." 
response_pattern = r"<\|assistant\|>\s*(.*)"


# granite vision
#device = "cuda" if torch.cuda.is_available() else "cpu"
# Verify CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("CUDA is not available. Using CPU")
    device = "cpu"

#device = "cpu"
#device = "cuda"

model_id = "ibm-granite/granite-vision-3.2-2b"
local_model_path = "/app/model-data/granite-vision-3.2-2b"

# Check if model exists locally and set the path accordingly
if os.path.exists(local_model_path) and os.path.isfile(os.path.join(local_model_path, "config.json")):
    print(f"Loading model from local volume: {local_model_path}")
    model_path = local_model_path
else:
    print(f"Downloading model from Hugging Face: {model_id}")
    print(f"Saving model to local volume: {local_model_path}")
    # Create directory if it doesn't exist
    os.makedirs(local_model_path, exist_ok=True)
    model_path = model_id

# Load model using the determined path
processor = AutoProcessor.from_pretrained(
    model_path, 
    use_fast=True,
    # this is useless and makes everything an oom slower
    #image_processor_kwargs={"do_resize": True, "size": {"height": 224, "width": 224}, "do_rescale": True, "rescale_factor": 1/255, "antialias": True}
)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    #low_cpu_mem_usage=True,
    #torch_dtype=torch.float32,
).to(device)

# Save model locally if we just downloaded it
if model_path == model_id:
    print(f"Saving model to local cache: {local_model_path}")
    processor.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)

if device == "cuda":
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory after loading:")
    print(f"  Allocated: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
    print(f"  Reserved: {reserved:.2f}GB")
    print(f"  Free for processing: {total - allocated:.2f}GB")

# Clear cache after loading
if device == "cuda":
    torch.cuda.empty_cache()



# gonna need to keep working on this to make it better
def process_codes(list_of_crops, case_index):
    total_time_start = time.time()
    
    # process all the images for vlm and ocr
    list_of_images = process_crops_for_vlm(list_of_crops, case_index)
    replaycodes = process_replaycodes_vlm(list_of_images, case_index)

    total_time = time.time() - total_time_start
    print(f"Total Time for VLM Test Suite: {total_time}")

    return replaycodes


# pre process all the crops into resized cv and PIL images
def process_crops_for_vlm(list_of_crops, case_index):
    list_of_images = []

    for code_index, crop in enumerate(list_of_crops):
        #h = 224
        h = 140
        w = int(h * 2.6)
        crop_resize = cv.resize(crop,(w, h))
        crop_pil = cv.cvtColor(crop_resize, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_pil)
        list_of_images.append((pil_image, crop_resize))
        if TEST:
            output_filename = f"/app/output/case{case_index}/crop{code_index}.png"
            cv.imwrite(output_filename, crop_resize)

    return list_of_images


def cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# maybe unbatch this?
def process_replaycodes_vlm(list_of_images, case_index):
    replaycodes = []

    # prepare batches of prompt + image to send to VLM
    batched = []
    for code_index, images in enumerate(list_of_images):
        pil_image = images[0]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        batched.append(conversation)

    # tokenize the batch
    inputs = processor.apply_chat_template(
        batched,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(device)

    # generate responses from batch
    vlm_outputs = model.generate(**inputs, max_new_tokens=25)
    for i in range(len(vlm_outputs)):
        full_response = processor.decode(
            vlm_outputs[i],
            skip_special_tokens=True,
            temperature=0.0, 
            do_sample=False
        )
        match = re.search(response_pattern, full_response, re.DOTALL)
        # is this enough?
        if match:
            code = match.group(1).strip()
        else:
            code = full_response.strip()

        code = code.upper()
        code = code.replace("O", "0")
        print(code)

        # default to ocr when not exactly 6 characters
        if len(code) != 6:
            crop = list_of_images[i][1]
            print(f"{case_index} {i}")
            code = ocr.process_code_mode2(crop, case_index, i)
        # not working? image is not good to be resized?
        #if len(code) != 6:
        #    crop = list_of_images[i][1]
        #    print(f"{case_index} {i}")
        #    code = ocr.process_code_mode1(crop, case_index, i)
        
        replaycodes.append(code)

    #del vlm_output, inputs, full_response, pil_image, crop_copy, crop_resize
    del vlm_outputs, batched, inputs, list_of_images
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    cleanup_cuda_memory()
    return replaycodes

