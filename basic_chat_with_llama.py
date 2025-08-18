from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json 



adapter_config_path = ".../adapter_config.json"

with open(adapter_config_path, 'r') as f:
    data = json.load(f)


base_model_path = "meta-llama/Llama-3.2-1B-Instruct"

# the file in which adapter file is downloaded
adapter_path = ".../phishing_chatbot"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map=device)
model = PeftModel.from_pretrained(base_model, adapter_path)

model.eval()

def chat():
    print("Time to chat :) Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        print("Model:", tokenizer.decode(outputs[0], skip_special_tokens=True))

chat()