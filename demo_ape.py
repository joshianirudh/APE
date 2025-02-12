import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama3-8b-instruct", "llama3.1-8b-instruct", "mistral-7b-instruct-v0.3", "gemma2-9b-it"])
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--scale", type=float, default=0.9)
    return parser.parse_args(args)

def load_model_and_tokenizer(model_name, device):
    if model_name == "llama3-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16).to(device)
    elif model_name == "llama3.1-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16).to(device)
    elif model_name == "mistral-7b-instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", torch_dtype=torch.bfloat16).to(device)
    elif model_name == "gemma2-9b-it":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.bfloat16).to(device)
    return tokenizer, model
        

def build_prefix(model_name, prompt):
    if "llama" in model_name:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    elif "mistral" in model_name:
        prompt = f"<s>[INST]{prompt}"
    elif "gemma" in model_name:
        prompt = f"<bos><start_of_turn>user\n{prompt}"
    return prompt

def build_suffix(model_name, prompt):
    if "llama" in model_name:
        prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "mistral" in model_name:
        prompt = f"{prompt}[/INST]"
    elif "gemma" in model_name:
        prompt = f"{prompt}<end_of_turn>\n<start_of_turn>model\n"   
    return prompt

def enable_attention_prefill_prefix(model_name, model):
    if "llama" in args.model:
        from src.ape_llama import enable_llama_attention_prefill_prefix
        enable_llama_attention_prefill_prefix(model)
    elif "mistral" in model_name:
        from src.ape_mistral import enable_mistral_attention_prefill_prefix
        enable_mistral_attention_prefill_prefix(model)
    elif "gemma" in model_name:
        from src.ape_gemma import enable_gemma_attention_prefill_prefix
        enable_gemma_attention_prefill_prefix(model)

def enable_attention_prefill_context(model_name, model):
    if "llama" in args.model:
        from src.ape_llama import enable_llama_attention_prefill_context
        enable_llama_attention_prefill_context(model)
    elif "mistral" in model_name:
        from src.ape_mistral import enable_mistral_attention_prefill_context
        enable_mistral_attention_prefill_context(model)
    elif "gemma" in model_name:
        from src.ape_gemma import enable_gemma_attention_prefill_context
        enable_gemma_attention_prefill_context(model)

def enable_attention_prefill_query(model_name, model, temperature, scale):
    if "llama" in args.model:
        from src.ape_llama import enable_llama_attention_prefill_query
        enable_llama_attention_prefill_query(model, temperature, scale)
    elif "mistral" in model_name:
        from src.ape_mistral import enable_mistral_attention_prefill_query
        enable_mistral_attention_prefill_query(model, temperature, scale)
    elif "gemma" in model_name:
        from src.ape_gemma import enable_gemma_attention_prefill_query
        enable_gemma_attention_prefill_query(model, temperature, scale)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def generate(args):
    prefix = ""
    contexts = [
        "My friends and I love going on road trips and exploring new places. However, we also enjoy hiking and camping together in nature.",
        "We often spend time playing board games and card games as a group. But we also like solving escape rooms and participating in trivia nights.",
        "Many of my friends enjoy listening to live music and attending concerts. We also love discovering new artists and sharing playlists with each other.",
        "We like trying out different coffee shops and bakeries. However, we also enjoy experimenting with baking and making homemade desserts.",
        "My friends and I love learning new skills, like photography and painting. We also enjoy visiting art workshops and DIY craft events."
    ]
    query = "Question: what are ten ideas for a social with a large groups of friends in New York City.\nAnswer:"



    device = torch.device(f'cuda:0')
    tokenizer, model = load_model_and_tokenizer(args.model, device)
    model = model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prefix = build_prefix(args.model, prefix)
    query = build_suffix(args.model, query)
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt").input_ids
        len_prefix = prefix_input_ids.shape[1]
        len_query = query_input_ids.shape[1]

        context_input_ids = tokenizer(contexts, return_tensors='pt', truncation=True, max_length=8192-len_prefix-len_query-256, padding=True, add_special_tokens=False).input_ids
        print(context_input_ids.shape)
        context_mask = (context_input_ids != tokenizer.pad_token_id).reshape(-1)
        
        enable_attention_prefill_prefix(args.model, model)
        past_key_values = None
        outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = []
        for past_key_value in outputs.past_key_values:
            bsz, _ = context_input_ids.shape
            past_key = past_key_value[0].repeat(bsz, 1, 1, 1)
            past_value = past_key_value[1].repeat(bsz, 1, 1, 1)
            past_position = past_key_value[2]
            past_key_values.append((past_key, past_value, past_position))

        enable_attention_prefill_context(args.model, model)
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

        enable_attention_prefill_query(args.model, model, args.temperature, args.scale)
        generation_kwargs = {}
        generation_kwargs["cache_implementation"] = None
        output = model.generate(
            input_ids=input_ids.to(model.device),
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            past_key_values=past_key_values,
            **generation_kwargs
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        print(pred)

if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)
    generate(args)
    