import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random

from ape import enable_llama_attention_prefill_prefix, enable_llama_attention_prefill_context, enable_llama_attention_prefill_query

def build_prefix(prompt):
    prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    return prompt

def build_suffix(prompt):
    prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def generate():
    prefix = ""
    contexts = [
        "My friends and I enjoy eating out at restaurants together. However, we also enjoy cooking and making food as a group as well."
        "Many of my friends like to play soccer and volleyball. We also enjoy watching movies and going to museums and galleries.",
    ]
    query = "Question: what are three ideas for a social with a large groups of friends in New York City.\nAnswer:"

    device = torch.device(f'cuda:0')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prefix = build_prefix(prefix)
    query = build_suffix(query)
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt").input_ids
        len_prefix = prefix_input_ids.shape[1]
        len_query = query_input_ids.shape[1]

        context_input_ids = tokenizer(contexts, return_tensors='pt', truncation=True, max_length=8192-len_prefix-len_query-256, padding=True, add_special_tokens=False).input_ids
        context_mask = (context_input_ids != tokenizer.pad_token_id).reshape(-1)
        len_context = (context_input_ids != tokenizer.pad_token_id).sum(dim=1)
            
        enable_llama_attention_prefill_prefix(model)
        past_key_values = None
        outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values_instruct = outputs.past_key_values
        past_key_values = []
        for past_key_value in outputs.past_key_values:
            bsz, _ = context_input_ids.shape
            past_key = past_key_value[0].repeat(bsz, 1, 1, 1)
            past_value = past_key_value[1].repeat(bsz, 1, 1, 1)
            past_position = past_key_value[2]
            past_key_values.append((past_key, past_value, past_position))

        enable_llama_attention_prefill_context(model)
        outputs = model(
            context_input_ids.to(model.device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = []

        for past_key_value in outputs.past_key_values:
            bsz, num_heads, seq_len, _ = past_key_value[0].size()
            past_key = torch.cat([past_key_value[0][:1, :, :len_prefix, :], 
                                    past_key_value[0][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)], dim=2)
            past_value = torch.cat([past_key_value[1][:1, :, :len_prefix, :], 
                                    past_key_value[1][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
            past_position = torch.cat([past_key_value[2][:, :len_prefix],
                                        past_key_value[2][:, len_prefix:].repeat(bsz, 1).flatten()[context_mask].unsqueeze(0)], dim=1)
            past_key_values.append((past_key, past_value, past_position, len(contexts)))
        context_input_ids = context_input_ids.flatten()[context_mask].unsqueeze(0)
        input_ids = torch.cat([prefix_input_ids, context_input_ids, query_input_ids], dim=-1)
        context_length = input_ids.shape[-1]

        enable_llama_attention_prefill_query(model, 1.0, 1.0)
        output = model.generate(
            input_ids=input_ids.to(model.device),
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            past_key_values=past_key_values,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        print(pred)

if __name__ == '__main__':
    seed_everything(42)
    generate()
    