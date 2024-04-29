import torch

from eagle.modeling_eagle import EAGLE
from transformers import AutoModelForCausalLM,AutoTokenizer

# base_model_path = "/lpai/volumes/cloudmodel-muses/lt/models/llava_qwen4b_sft_v4.5"
base_model_path = "/mnt/volumes/cloudmodel-muses/lt/models/Llama-2-7b-chat-hf"
# eagle_path = "/lpai/EAGLE/checkpoints_20/model_20/"
eagle_path = "/mnt/volumes/cloudmodel-muses/lt/models/EAGLE-llama2-chat-7B"

tokenizer=AutoTokenizer.from_pretrained(base_model_path)
model=AutoModelForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16,device_map="cuda",)

text="hi"
text = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": text}
]
text = tokenizer.apply_chat_template(
    text,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt",padding=False)

eagle=EAGLE(model,eagle_path)
print(eagle.base_model)
print(eagle.ea_layer)
print(eagle.base_model.device)
print(eagle.ea_layer.device)
inputs["input_ids"].cuda()
inputs["attention_mask"].cuda()

outs=eagle.generate(**inputs, max_new_tokens=200)
print("outs", outs)
output=tokenizer.decode(outs)
print("output", output)