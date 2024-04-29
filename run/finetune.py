from eagle.modeling_eagle import EAGLE
from transformers import AutoModelForCausalLM,AutoTokenizer

base_model_path = "/mnt/volumes/cloudmodel-muses/lt/models/llava_qwen4b_sft_v4.5"

tokenizer=AutoTokenizer.from_pretrained(base_model_path)
model=AutoModelForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16,device_map="cuda",)
# for bs>1, the padding side should be right
if bs>1:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

text=prompt1
# text=[prompt1,prompt2]
inputs = tokenizer(text, return_tensors="pt",padding=True)

eagle=EAGLE(model,eagle_path)
outs=eagle.generate(**inputs, max_new_tokens=200,temperature=0.0)
output=tokenizer.decode(outs)
# output=tokenizer.batch_decode(outs)