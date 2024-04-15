import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/external/models/llama-2-7b-hf/"
tokenizer_name = "/external/models/llama-2-7b-hf/"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "How to keep fit"

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
    )

generated_text_ids = outputs[0]

generated_text = tokenizer.decode(generated_text_ids, skip_special_tokens=True)
print(f"Generated text: {generated_text}")