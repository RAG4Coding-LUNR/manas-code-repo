from transformers import pipeline

question = "Write a program to add numbers together"
generator = pipeline("text-generation", model="./qwen2.5-finetuned/checkpoint-500", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])