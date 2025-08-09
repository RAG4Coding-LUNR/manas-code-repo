from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import time

import os

dir = "/home/avisingh/models/codellama-RAG-v5 -wo-chat"

os.environ["WANDB_PROJECT"] = "Unsloth-CodeLlama-RAG"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


# 1. Training configuration
max_seq_length = 8192 # Set context length to 4096, changing it to 5120 for a quick experiment

# 2. Load Qwen 2.5 in 16-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/codellama-7b",
    max_seq_length=max_seq_length,
    load_in_4bit=False, # Switched from 4-bit to 16-bit precision for Bits&Bytes.
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16, # Explicitly set the data type to 16-bit. bfloat16 is used if available for better performance, otherwise float16.
)

# 3. Wrap with LoRA adapters (PEFT)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    # CHANGE: Target modules are set to train all linear layers, which is the
    # default for Qwen2 in Unsloth. The list below includes all of them.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_rslora=False,
    use_gradient_checkpointing="unsloth",
)

# 4. Load & preprocess your CSV dataset
dataset = load_dataset(
    "csv",
    data_files={"train": "/home/avisingh/datasets/training_data_v2.csv"},
    split="train",
)

# CHANGE: Randomly sample 10,000 data points from the dataset. NOT NEEDED, ALREADY SAMPLED
# dataset = dataset.shuffle(seed=42).select(range(6000))
dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

# 5. Apply the llama chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama") # Only for EOS token

'''
def apply_prompt_response_template(examples):
    prompts = examples["Prompt"]
    contexts = examples["Context"]
    responses = examples["Target"]
    # build Unsloth-style conversation
    convos = [
        [
            {"role": "user", "content": f"{c}\n Based on the context provided, complete the prompt: {p}\n "},
            {"role": "assistant", "content": r}
        ]
        for p, c, r in zip(prompts, contexts, responses)
    ]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}
'''

# Testing out wo a chat template
def apply_prompt_response_template(examples):
    prompts = examples["Prompt"]
    contexts = examples["Context"]
    responses = examples["Target"]
    
    eos_token = tokenizer.eos_token 

    texts = []
    for prompt, context, response in zip(prompts, contexts, responses):
        text = (
            #f"Based on the context below, complete the prompt.\n\n"
            f"Context:\n{context}\n"
            f"Prompt:\n{prompt}\n"
            f"Code:\n{response}{eos_token}" # Add the EOS token here!
        )
        texts.append(text)
        
    return {"text": texts}

train_dataset = train_dataset.map(
    apply_prompt_response_template,
    batched=True,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    apply_prompt_response_template,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

# 6. Set up the TRL SFT trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
    args=TrainingArguments(
        learning_rate=5e-5, # Learning rate is set to 1e-4, changing it to 2e-5 to see if it helps with overfitting
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1, # Batch size is set to 1.
        gradient_accumulation_steps=32,  # Gradient accumulation is set to 32 steps.
        num_train_epochs=2, # Number of training epochs is set to 3, changing it to 1 to see if it helps with overfitting
        fp16=not is_bfloat16_supported(), # Precision is set to 16-bit (bf16 or fp16).
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.1,
        output_dir=dir,
        seed=42,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=100,
        eval_steps=50,
        report_to="wandb",
        run_name="CodeLlama-7b-RAG-v5-wo-chat"
    ),
)

print("Starting fine-tuning: Unsloth/Codellama-7b...")
trainer.train(resume_from_checkpoint=False)
print("Training complete.")

trainer.save_model(output_dir=dir)

print("Model Saved")