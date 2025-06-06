def process_replaycodes_vlm(list_of_images, case_index):
    if not list_of_images:
        return []

    replaycodes = []

    # Process images one by one
    for code_index, images in enumerate(list_of_images):
        pil_image = images[0]
        
        try:
            with torch.no_grad():
                # Method 1: Try direct processing without conversation format
                # This avoids the multi-patch processing that creates extra dimensions
                inputs = processor(
                    text=prompt,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to device and handle dtypes carefully
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Handle pixel_values dtype for quantized model
                if "pixel_values" in inputs:
                    # Check and fix shape if needed
                    if len(inputs["pixel_values"].shape) == 5 and inputs["pixel_values"].shape[1] > 1:
                        print(f"Fixing pixel_values shape from {inputs['pixel_values'].shape}")
                        # Take the first patch/view only
                        inputs["pixel_values"] = inputs["pixel_values"][:, 0, :, :, :]
                    
                    # Convert to model's expected dtype
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
                    
                    print(f"Image {code_index} - pixel_values shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")
                
                # Ensure input_ids are in the correct dtype for quantized model
                if "input_ids" in inputs:
                    inputs["input_ids"] = inputs["input_ids"].to(torch.long)
                
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
                
                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=25,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
                # Decode the response
                generated_text = processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # Extract code from the response
                # Remove the prompt part if it exists
                if prompt in generated_text:
                    code = generated_text.split(prompt)[-1].strip()
                else:
                    # Look for patterns that might indicate the answer
                    # Sometimes models add extra text before/after
                    code = generated_text.strip()
                    # Try to extract just the code part
                    lines = code.split('\n')
                    for line in lines:
