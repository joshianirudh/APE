import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
import copy
import math

from ape import enable_llama_attention_prefill_prefix, enable_llama_attention_prefill_context, enable_llama_attention_prefill_query

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "llama3-8b-instruct", "llama3.1-8b-instruct", "mistral-7b", "gemma"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        prompt = f"<s>[INST]{prompt}[/INST]"
    elif "gemma" in model_name:
        prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif "llama3" in model_name:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def build_prefix(tokenizer, prompt, model_name):
    if "mistral" in model_name:
        prompt = f"<s>[INST]{prompt}"
    elif "gemma" in model_name:
        prompt = f"<bos><start_of_turn>user\n{prompt}"
    elif "llama3" in model_name:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    return prompt

def build_suffix(tokenizer, prompt, model_name):
    if "mistral" in model_name:
        prompt = f"{prompt}[/INST]"
    elif "gemma" in model_name:
        prompt = f"{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif "llama3" in model_name:
        prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def split_tensor(tensor, split_size):
    # Get the size of the first dimension
    first_dim_size = tensor.size(0)
    
    # Calculate the number of splits
    num_splits = (first_dim_size + split_size - 1) // split_size
    
    # Use torch.split to divide the tensor
    splits = torch.split(tensor, split_size)
    
    # Convert each split tensor to a list
    result = list(splits)
    
    return result


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        if json_obj["length"] < 8000:
            continue
        prompt = prompt_format.format(**json_obj)
        instruction = ""
        document = ""
        query = ""
        prompt_split = prompt.split("\n\r\r\r")
        assert len(prompt_split) == 3
        #print(prompt_split)
        #instruction = prompt_split[0]
        document = prompt_split[1]
        query = prompt_split[0] + prompt_split[2]
        if dataset not in ["lcc", "repobench-p"]:
            instruction = build_prefix(tokenizer, instruction, model_name)
            query = build_suffix(tokenizer, query, model_name)
        with torch.no_grad():
            instruction_input_ids = tokenizer(instruction, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
            query_input_ids = tokenizer(query, truncation=False, return_tensors="pt").input_ids
            len_instruction = instruction_input_ids.shape[1]
            len_query = query_input_ids.shape[1]
            document_input_ids = tokenizer(document, add_special_tokens=False, return_tensors='pt').input_ids
            len_document = len(document_input_ids[0])
            length = 7500
            n = math.ceil(len(document_input_ids[0]) / length)
            length = (len_document + (n-1)) // n
            insert_idx = (n+1) // 2
            padding = torch.ones(size=(1, n * length - len(document_input_ids[0])), dtype=document_input_ids.dtype, device=document_input_ids.device) * tokenizer.pad_token_id
            document_input_ids = torch.cat([document_input_ids[:, :-(n - insert_idx) * length], padding, document_input_ids[:, -(n - insert_idx) * length:]], dim=-1)
            documents_input_ids = document_input_ids.reshape(-1, length)
            
            enable_llama_attention_prefill_prefix(model)
            past_key_values = None
            outputs = model(
                instruction_input_ids.to(model.device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values_instruct = outputs.past_key_values
            past_key_values = list(outputs.past_key_values)
            input_ids = instruction_input_ids
            documents = split_tensor(documents_input_ids, 5)

            enable_llama_attention_prefill_context(model)
            for document_input_ids in documents:
                document_mask = (document_input_ids != tokenizer.pad_token_id).reshape(-1)
                len_document = (document_input_ids != tokenizer.pad_token_id).sum(dim=1)
                temperature = torch.cat([torch.full((int(length),), len_document.sum()/(length * len_document.shape[0])) for length in len_document])
                past_key_values_local = []
                for past_key_value in past_key_values_instruct:
                    bsz, _ = document_input_ids.shape
                    past_key = past_key_value[0].repeat(bsz, 1, 1, 1)
                    past_value = past_key_value[1].repeat(bsz, 1, 1, 1)
                    past_position = past_key_value[2]
                    past_key_values_local.append((past_key, past_value, past_position))
                outputs = model(
                    document_input_ids.to(model.device),
                    past_key_values=past_key_values_local,
                    use_cache=True,
                )
                past_key_values_local = outputs.past_key_values
                input_ids = torch.cat([input_ids, document_input_ids.flatten()[document_mask].unsqueeze(0)], dim=-1)
                past_key_values_new = []
                for past_key_value_global, past_key_value_local in zip(past_key_values, past_key_values_local):
                    #print(past_key_value_local[0].shape, past_key_value_global[0].shape, document_mask.shape)
                    bsz, num_heads, seq_len, _ = past_key_value_local[0].size()
                    past_key = torch.cat([past_key_value_global[0], 
                                          past_key_value_local[0][:, :, len_instruction:, :].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                    past_value = torch.cat([past_key_value_global[1], 
                                            past_key_value_local[1][:, :, len_instruction:, :].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2) 
                    past_position = torch.cat([past_key_value_global[2],
                                               past_key_value_local[2][:, len_instruction:].repeat(bsz, 1).flatten()[document_mask].unsqueeze(0)], dim=1)
                    past_key_values_new.append((past_key, past_value, past_position))
                past_key_values = past_key_values_new
            past_key_values_new = []
            for past_key_value in past_key_values:
                past_key_values_new.append((past_key_value[0], past_key_value[1], past_key_value[2], True))
            del past_key_values
            del past_key_values_instruct
            del past_key_values_local
            del outputs
            input_ids = torch.cat([input_ids, query_input_ids], dim=-1)
            context_length = input_ids.shape[-1]
            generation_kwargs = {}
            #generation_kwargs["cache_implementation"] = None
            enable_llama_attention_prefill_query(model)
            output = model.generate(
                input_ids=input_ids.to(model.device),
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                past_key_values=past_key_values_new,
                **generation_kwargs
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        #replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        from modeling_llama_parallel import enable_llama_parallel_attention
        enable_llama_parallel_attention(model)
    elif "llama3" in model_name:
        #replace_llama_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        #from modeling_llama_parallel import enable_llama_parallel_attention
        #enable_llama_parallel_attention(model)
    elif "mistral" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        from modeling_mistral_parallel import enable_mistral_parallel_attention
        enable_mistral_parallel_attention(model)
    elif "gemma" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        from modeling_gemma_parallel import enable_gemma_parallel_attention
        enable_gemma_parallel_attention(model)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        #datasets = ["qasper", "multifieldqa_en", "gov_report", "samsum"]
        datasets = ["gov_report", "multi_news"]
    else:
        #datasets = ["multi_news"]
        datasets = ["narrativeqa", "qmsum", "musique"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt3.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred_parallel"):
        os.makedirs("pred_parallel")
    if not os.path.exists("pred_parallel_e"):
        os.makedirs("pred_parallel_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_parallel_e/{model_name}"):
                os.makedirs(f"pred_parallel_e/{model_name}")
            out_path = f"pred_parallel_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred_parallel/{model_name}"):
                os.makedirs(f"pred_parallel/{model_name}")
            out_path = f"pred_parallel/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
