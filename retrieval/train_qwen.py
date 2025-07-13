from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


# 1. Training configuration
max_seq_length = 2048 # Set context length to 2048

# 2. Load Qwen 2.5 in 16-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=False, # Switched from 4-bit to 16-bit precision for Bits&Bytes.
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16, # Explicitly set the data type to 16-bit. bfloat16 is used if available for better performance, otherwise float16.
)

# 3. Wrap with LoRA adapters (PEFT)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=32,
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
    data_files={"train": "training_data_sampled.csv"},
    split="train",
)

# CHANGE: Randomly sample 10,000 data points from the dataset. NOT NEEDED, ALREADY SAMPLED
# dataset = dataset.shuffle(seed=42).select(range(10000))


# 5. Apply the Qwen-2.5 chat template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def apply_prompt_response_template(examples):
    prompts = examples["Prompt"]
    contexts = examples["Context"]
    responses = examples["Target"]
    # build Unsloth-style conversation
    convos = [
        [
            {"from": "user", "value": f"{p}\nHere is some retrieved context to help solve this problem:\n{c}"},
            {"from": "assistant", "value": r}
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

dataset = dataset.map(
    apply_prompt_response_template,
    batched=True,
    remove_columns=dataset.column_names,
)

# 6. Set up the TRL SFT trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
    args=TrainingArguments(
        learning_rate=2e-5, # Learning rate is set to 1e-4, changing it to 2e-5
        lr_scheduler_type="linear",
        per_device_train_batch_size=1, # Batch size is set to 1.
        gradient_accumulation_steps=32,  # Gradient accumulation is set to 32 steps.
        num_train_epochs=1, # Number of training epochs is set to 3, changing it to 1
        fp16=not is_bfloat16_supported(), # Precision is set to 16-bit (bf16 or fp16).
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="./qwen2.5-coder-finetuned",
        seed=42,
        save_strategy="steps",
        save_steps=100
    ),
)

print("Starting fine-tuning: Qwen/Qwen2.5-7B with Qwen's native chat template...")
trainer.train(resume_from_checkpoint=False)
print("Training complete.")

trainer.save_model(output_dir="./qwen2.5-finetuned")

print("Model Saved")