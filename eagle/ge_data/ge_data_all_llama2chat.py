import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
import llava
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.eval.llava_mixtral_eval import create_data_loader, load_pretrained_model
# from llava.train.train import DataArguments, LazySupervisedDataset
from llava.model import LlavaQwenForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template

# bigname="/lpai/volumes/cloudmodel-muses/lt/models/llava_qwen4b_sft_v4.5"
bigname="/mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-04-28-01"
# bigname = "/home/lyh/weights/hf/llama/7B/"
# smallname = "/home/lyh/weights/hf/llama/7B/"
data_file = "/mnt/volumes/cloudmodel-muses/yjiang/data/VLM/03_ExpData/SFT/Release/v0.4.5_yq/driving/drivelm_zh_valid.json"

def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('json', data_files=data_file)
    ds = ds['train']
    # ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            conv = get_conversation_template("llama-2-chat")
            sys_p="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message=sys_p
            roles = {"user": conv.roles[0], "assistant": conv.roles[1]}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                if sentence["from"]=="assistant":
                    sentence["value"]=" "+sentence["value"]
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()
            # if i==56:
            #     print(i)
            # if i==57:
            #     print(i)
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)
            #print(i)

            sep = conv.sep + conv.roles[1] + " "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     instruction_len -= 1

                # Ignore the user instructions
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                    # # The legacy and non-legacy modes handle special tokens differently
                    # cur_len -= 1

            loss_mask[cur_len:] = 0


            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1

# bigtokenizer = AutoTokenizer.from_pretrained(bigname,use_fast=False)
# ds = build_dataset_rank(bigtokenizer)
# print(ds)
# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname, load_in_4bit=True, device_map={"": 0}, )
# smallmodel = AutoModelForCausalLM.from_pretrained(smallname, load_in_4bit=True, device_map={"": 1}, )
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",load_in_8bit=True)
# bigmodel.eval()


with open(data_file, 'r') as f:
    questions = json.load(f)
    questions = questions[args.start: args.end]

tokenizer, bigmodel, image_processor, context_len = load_pretrained_model(
            bigname, model_base="llava_qwen", load_in_8bit=False, load_in_4bit=False)
ds = create_data_loader(
    questions,
    "/mnt/volumes/cloudmodel-muses/llava_data",
    tokenizer,
    image_processor,
    bigmodel.config,
    batch_size=1
)

# for idx,data in enumerate(ds):
    # input_ids, image_tensor, image_sizes, prompt = data
    # loss_mask=torch.ones_like(input_ids)
    # prompt = prompt[0]
    # print("input_ids:", input_ids)
    # print("loss_mask:", loss_mask)
    # print("prompt:", prompt)
    # for ele in input_ids[0]:
        # print("prompt decode:", tokenizer.decode(ele))
    # print("prompt decode:", tokenizer.batch_decode(input_ids))
    # for chunk in prompt.split('<image>'):
        # print("chunk:", chunk)   
        # print(tokenizer(chunk).input_ids)
    # if idx == 10:
        # exit()


@torch.no_grad()
def ge(data):
    (input_ids, image_tensor, image_sizes, prompt) = data

    input_ids = input_ids.to(device='cuda', non_blocking=True)

    with torch.inference_mode():
        # output_ids = bigmodel.generate(
            # input_ids,
            # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            # image_sizes=image_sizes,
            # do_sample=True if args.temperature > 0 else False,
            # temperature=args.temperature,
            # top_p=args.top_p,
            # num_beams=args.num_beams,
            # max_new_tokens=args.max_new_tokens,
            # use_cache=False)

        (
            inputs,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _
        ) = bigmodel.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=image_sizes
        )

        outs_big = bigmodel(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )

    image_len = inputs_embeds.shape[1] - input_ids.shape[1]
    loss_mask=torch.zeros(inputs_embeds.shape[0],inputs_embeds.shape[1])
    mask = 0
    for idx in range(input_ids[0].numel()):
        loss_mask[0][image_len+idx] = mask
        if (input_ids[0][idx] == 60):
            if (input_ids[0][idx-1] == 64462):
                if (input_ids[0][idx-2] == 64928):
                    mask = 1
        if (mask == 1) and (input_ids[0][idx] == 151643):
            mask = 0
        
    # outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp=probs[0].max(dim=1).values
    td={"input_ids":input_ids.cpu()[0],"inputs_embeds":inputs_embeds.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":loss_mask.cpu()[0]}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for id,data in enumerate(ds):
    if id%100==0:
        print(id,end="\t")
    if id % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir,outdata)


