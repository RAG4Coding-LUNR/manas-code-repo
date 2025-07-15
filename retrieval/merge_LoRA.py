import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os

# --- Configuration ---
# NOTE: We use the original Hugging Face model name here, NOT the unsloth version.
# This is crucial for using the standard transformers library correctly.
base_model_name = "unsloth/codellama-7b" 

# The path to your saved LoRA adapter
adapter_path = "./qwen2.5-finetuned"

# The directory to save the final, correct model
output_path = "./qwen2.5-coder-CAFT"

print("--- Starting Robust Merge Process ---")
print(f"This will be slower and use more memory, but is more reliable.")

# --- 1. Load Model and Tokenizer using standard Transformers ---
print(f"\n[1/5] Loading base model '{base_model_name}'...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"[2/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --- 2. Apply LoRA adapter ---
print(f"[3/5] Applying adapter from '{adapter_path}'...")
merged_model = PeftModel.from_pretrained(base_model, adapter_path)

# --- 3. Merge the adapter into the model ---
print(f"[4/5] Merging adapter weights...")
merged_model = merged_model.merge_and_unload()

# --- 4. Save the final model ---
print(f"[5/5] Saving final model to '{output_path}'...")
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("\n✅ PROCESS COMPLETE. The model in '{output_path}' should now be correct.")