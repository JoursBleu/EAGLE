import argparse
import torch
import time

from datasets import load_dataset
from eagle.model.ea_model import EaModel
from eagle.model.modeling_llama_kv import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HfLlamaForCausalLM

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--small_model_dir', type=str, default=None)
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument('--run_hf', action="store_true", default=False)
    parser.add_argument('--use_sp', action="store_true", default=False)
    parser.add_argument('--benchmark', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_steps', type=int, default=10)
    parser.add_argument('--benchmark_dataset', type=str, default="ccdv/cnn_dailymail")
    parser.add_argument('--benchmark_dataset_version', type=str, default="3.0.0")
    parser.add_argument('--benchmark_dataset_json', type=str, default=None)

    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
print("loading model")
if (args.run_hf):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="cuda"
    ).eval()
    print(model.dtype)
else:
    model = EaModel.from_pretrained(
        base_model_path=args.base_model_dir,
        ea_model_path=args.small_model_dir,
        torch_dtype=torch.float16,
        load_in_4bit=False,
    ).eval().cuda()

# print("compiling...")
# model.draft_one=torch.compile(model.draft_one, mode="reduce-overhead", fullgraph=True,dynamic=True)
# model.base_forward=torch.compile(model.base_forward, mode="reduce-overhead", fullgraph=True,dynamic=True)
# model.base_forward_one=torch.compile(model.base_forward_one, mode="reduce-overhead", fullgraph=True)
# print("done compile.")


total_time = 0
total_token = 0
avg_acc_lens = 0
data_id = 55000
if (args.benchmark_dataset_json is not None):
    dataset = load_dataset('json', data_files=args.benchmark_dataset_json)['train']
else:
    dataset = load_dataset(args.benchmark_dataset, args.benchmark_dataset_version, split="test")
count = 0
avg_acc_len = 0
total_input_len = 0

i=0
# while count < args.benchmark_steps:
while data_id < 55000 + 100:
    batch = []
    while(len(batch) < args.batch_size):
        text = dataset[data_id]['conversations'][0]['value']
        text = [
            {"role": "system", "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            {"role": "user", "content": text}
        ]
        seq = tokenizer.apply_chat_template(
            text,
            tokenize=False,
            add_generation_prompt=True
        )
        print("seq", seq)
        data_id += 1
        token_ids = torch.squeeze(tokenizer(seq, return_tensors="pt")["input_ids"])
        # if  token_ids.shape[0] >= args.max_input_len:
            # inputs = token_ids[0:args.max_input_len]
            # batch.append(inputs)
            # input_len = args.max_input_len
        input_len = len(token_ids)
        batch.append(token_ids)
        # print("token_ids.shape[0]", token_ids.shape[0])
        # batch.append(token_ids)
        # input_len = token_ids.shape[-1]
    if len(batch) > 1:
        batch = torch.concat(batch).reshape(args.batch_size, input_len)
    else:
        batch = batch[0].reshape(args.batch_size, input_len)
    print("intput", batch)
    start = time.time()
    if (args.run_hf):
        outputs = model.generate(batch.cuda(), max_new_tokens=args.max_new_tokens)
    elif (args.use_sp):
        outputs, avg_acc_len = model.eagenerate(batch.cuda(), max_new_tokens=args.max_new_tokens)
        # outputs, avg_acc_len = model.eagenerate(batch.cuda(), top_p=0.9, temperature=0.5, max_new_tokens=args.max_new_tokens)
    else:
        outputs = model.naivegenerate(batch.cuda(), max_new_tokens=args.max_new_tokens)
        # outputs = model.naivegenerate(batch.cuda(), top_p=0.9, temperature=0.5, max_new_tokens=args.max_new_tokens)
    end = time.time()
    print("Step", i, ":", end - start, "s")
    print("Outputs:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    new_token = (outputs.numel() - input_len * args.batch_size)
    print("Total new token:", new_token)
    print("Avg acc len:", avg_acc_len)

    # if ((i > 1) and (new_token>=args.max_new_tokens)):
    # out = outputs[input_len:]
    # val = False
    # for ele in out:
        # if ele != 
    if (i > 1):
        count += 1
        total_time += (end - start)
        total_token += new_token
        avg_acc_lens += avg_acc_len
        total_input_len += input_len
        print("avg_acc_lens:", avg_acc_lens / (i+1))
    i += 1
print("Total steps:", args.benchmark_steps)
print("Batch size:", args.batch_size)
print("Input len:", total_input_len / count)
print("Avg output len:", total_token / count)
print("Avg acc len:", avg_acc_lens / count)
print("benchmark_steps:", count)
print("TPS:", total_token / total_time)

