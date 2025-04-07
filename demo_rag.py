import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
import random
import argparse

from ape import enable_attention_prefill_prefix, enable_attention_prefill_context, enable_attention_prefill_query

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        choices=["llama3-8b-instruct", "llama3.1-8b-instruct",
                                 "mistral-7b-instruct-v0.3", "gemma2-9b-it"])
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--scale", type=float, default=0.9)
    parser.add_argument("--num_ctx", type=int, default=5, help="Number of retrieved contexts to use.")
    parser.add_argument("--arguana_path", type=str, default="arguana/128k/corpus.jsonl",
                        help="Path to the ArguAna 128k corpus file (one passage per line).")
    return parser.parse_args(args)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name, device):
    """
    Loads the user-chosen LLM and tokenizer in 8-bit floating.
    """
    if model_name == "llama3-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                     torch_dtype=torch.bfloat16).to(device)
    elif model_name == "llama3.1-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                                     torch_dtype=torch.bfloat16).to(device)
    elif model_name == "mistral-7b-instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                                     torch_dtype=torch.bfloat16).to(device)
    elif model_name == "gemma2-9b-it":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it",
                                                     torch_dtype=torch.bfloat16).to(device)
    return tokenizer, model

def build_prefix(model_name, prompt):
    """
    Builds any necessary system tokens or prefix based on model naming.
    """
    if "llama" in model_name:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    elif "mistral" in model_name:
        prompt = f"<s>[INST]{prompt}"
    elif "gemma" in model_name:
        prompt = f"<bos><start_of_turn>user\n{prompt}"
    return prompt

def build_suffix(model_name, prompt):
    """
    Builds suffix tokens for different model types.
    """
    if "llama" in model_name:
        prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "mistral" in model_name:
        prompt = f"{prompt}[/INST]"
    elif "gemma" in model_name:
        prompt = f"{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return prompt

def retrieve_contexts(args, question):
    """
    Minimal retrieval pipeline using Contriever on an ArguAna 128k corpus.
    Loads passages from a file (one per line). Returns top-N contexts.
    """
    # Load Contriever for retrieval
    contriever_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    query_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()
    context_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()

    # Read all lines from ArguAna 128k corpus
    with open(args.arguana_path, "r", encoding="utf-8") as f:
        all_passages = [line.strip() for line in f if line.strip()]

    # Form a single query
    question_input = contriever_tokenizer(question, return_tensors='pt').input_ids.cuda()
    ctx_input = contriever_tokenizer(all_passages, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.cuda()

    with torch.inference_mode():
        query_emb = query_encoder(question_input).last_hidden_state[:, 0, :]
        ctx_emb = context_encoder(ctx_input).last_hidden_state[:, 0, :]
        similarities = query_emb.matmul(ctx_emb.transpose(0, 1))

    # Sort passages by similarity descending, keep top-n
    indices = torch.argsort(similarities, dim=-1, descending=True)
    top_ctx = [all_passages[i] for i in indices[0][: args.num_ctx]]
    return top_ctx

def generate(args):
    device = torch.device('cuda:0')

    # ============ Single sample query ============
    question = "What are some arguments to support universal basic income?"
    # ---------------------------------------------

    # retrieve top contexts from arguana 128k
    contexts = retrieve_contexts(args, question)

    # Build query
    query = f"Question: {question}\nAnswer:"
    prefix = ""  # can keep empty or add minimal prefix

    # Load main LLM
    tokenizer, model = load_model_and_tokenizer(args.model, device)
    model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply APE's recommended prefix and suffix
    prefix = build_prefix(args.model, prefix)
    query = build_suffix(args.model, query)

    with torch.no_grad():
        # Convert prefix & query to token IDs
        prefix_input_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, return_tensors="pt").input_ids

        len_prefix = prefix_input_ids.shape[1]
        len_query = query_input_ids.shape[1]

        # Convert contexts to token IDs (batched). We can do minimal truncation if needed.
        context_input_ids = tokenizer(contexts, return_tensors='pt',
                                      truncation=True,
                                      max_length=8192 - len_prefix - len_query - 256,
                                      padding=True,
                                      add_special_tokens=False).input_ids

        context_mask = (context_input_ids != tokenizer.pad_token_id).reshape(-1)

        # 1) Encode prefix using prefix attention
        enable_attention_prefill_prefix(args.model, model)
        outputs = model(prefix_input_ids.to(model.device), use_cache=True)
        past_key_values = outputs.past_key_values

        # Repeat the prefix's PKV across the batch dimension for context
        new_past_key_values = []
        bsz, _ = context_input_ids.shape
        for pkv in past_key_values:
            past_key, past_value, past_position = pkv
            # repeat prefix KV states for each context batch
            rep_key = past_key.repeat(bsz, 1, 1, 1)
            rep_value = past_value.repeat(bsz, 1, 1, 1)
            new_past_key_values.append((rep_key, rep_value, past_position))
        past_key_values = new_past_key_values

        # 2) Encode contexts in parallel
        enable_attention_prefill_context(args.model, model)
        outputs = model(
            context_input_ids.to(model.device),
            past_key_values=past_key_values,
            use_cache=True,
        )

        # gather updated PKVs
        past_key_values = []
        for pkv in outputs.past_key_values:
            # pkv shape: (batch_size, n_heads, seq_len, head_dim)
            # need to unify prefix & masked context
            bsz, num_heads, seq_len, _ = pkv[0].size()
            prefix_len = len_prefix
            k, v, p_pos = pkv
            # separate prefix part from context part, flatten & keep only mask tokens
            new_k = torch.cat([
                k[:1, :, :prefix_len, :],  # prefix for a single batch
                k[:, :, prefix_len:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)
            ], dim=2)
            new_v = torch.cat([
                v[:1, :, :prefix_len, :],
                v[:, :, prefix_len:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)
            ], dim=2)

            new_pos = torch.cat([
                p_pos[:, :prefix_len],
                p_pos[:, prefix_len:].repeat(bsz, 1).flatten()[context_mask].unsqueeze(0)
            ], dim=1)

            past_key_values.append((new_k, new_v, new_pos, len(contexts)))

        # re-construct final input_ids
        context_input_ids = context_input_ids.flatten()[context_mask].unsqueeze(0)
        input_ids = torch.cat([prefix_input_ids, context_input_ids, query_input_ids], dim=-1)
        context_length = input_ids.shape[-1]

        # 3) final query + generation w/ APE query setting
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
