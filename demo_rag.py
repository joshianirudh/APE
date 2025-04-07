import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
import argparse
from ape import enable_attention_prefill_prefix, enable_attention_prefill_context, enable_attention_prefill_query
from transformers import AutoTokenizer, AutoModel
import faiss

def load_corpus(file_path):
    """Load documents from corpus.jsonl."""
    with open(file_path, 'r') as f:
        corpus = [json.loads(line)['text'] for line in f]
    return corpus

def load_queries(file_path):
    """Load queries from a query file (e.g., dev_queries.jsonl)."""
    with open(file_path, 'r') as f:
        queries = [json.loads(line)['query'] for line in f]
    return queries

def build_index_from_corpus(corpus, tokenizer, model):
    """Build a FAISS index from the given corpus."""
    inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_top_k(query, index, corpus, tokenizer, model, k=5):
    """Retrieve top-k relevant documents for a query."""
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    distances, indices = index.search(query_embedding, k)
    return [corpus[i] for i in indices[0]]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama3-8b-instruct", "llama3.1-8b-instruct"])
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--scale", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=5)  # Number of top documents to retrieve
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
    if model_name == "llama3-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16).to(device)
    elif model_name == "llama3.1-8b-instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16).to(device)
    return tokenizer, model

def build_prefix(model_name, prompt):
    if "llama" in model_name:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    return prompt

def build_suffix(model_name, prompt):
    if "llama" in model_name:
        prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt


def setup_contriever():
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")
    return tokenizer, model

def build_index(documents, tokenizer, model):
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_top_k(query, index, documents, tokenizer, model, k=5):
    query_embedding = model(**tokenizer([query], return_tensors="pt")).last_hidden_state.mean(dim=1).detach().numpy()
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# ---------------------- Main Generation Logic ----------------------

def generate(args):
    # Setup device and load models
    device = torch.device(f'cuda:0')
    tokenizer_llama, model_llama = load_model_and_tokenizer(args.model, device)
    tokenizer_contriever, model_contriever = setup_contriever()

    # Load ArguAna-128k dataset
    corpus_path = "arguana/128k/corpus.jsonl"
    queries_path = "arguana/128k/dev_queries.jsonl"
    
    # Load corpus and queries
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)

    # Build FAISS index from corpus
    index = build_index_from_corpus(corpus, tokenizer_contriever, model_contriever)

    # Retrieve top-k contexts using Contriever
    sample_query = queries[0]
    top_k_documents = retrieve_top_k(sample_query, index, corpus, tokenizer_contriever, model_contriever, k=args.top_k)

    # Tokenize prefix and query
    prefix_input_ids = tokenizer_llama(build_prefix(args.model, ""), truncation=False,
                                       return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    query_input_ids = tokenizer_llama(build_suffix(args.model, sample_query), truncation=False,
                                      return_tensors="pt").input_ids.to(device)

    # Tokenize retrieved contexts
    context_input_ids = tokenizer_llama(top_k_documents,
                                        return_tensors='pt', truncation=True,
                                        max_length=8192 - prefix_input_ids.shape[1] - query_input_ids.shape[1] - 256,
                                        padding=True,
                                        add_special_tokens=False).input_ids.to(device)

    # Enable APE attention mechanisms for prefix
    enable_attention_prefill_prefix(args.model, model_llama)
    
    # Compute KV states for prefix and context as before...
    with torch.no_grad():
        outputs_prefix = model_llama(prefix_input_ids, use_cache=True)
        past_key_values_prefix = outputs_prefix.past_key_values

    # Enable APE attention mechanisms for context
    enable_attention_prefill_context(args.model, model_llama)
    
    # Compute KV states for contexts
    outputs_context = model_llama(
        context_input_ids,
        past_key_values=past_key_values_prefix,
        use_cache=True
    )
    past_key_values_context = outputs_context.past_key_values
    
    # Store KV states to CPU
    context_kv_cpu = [(k.cpu(), v.cpu()) for k, v in past_key_values_context]
    
    # Load back to GPU when needed
    context_kv_gpu = [(k.cuda(), v.cuda()) for k, v in context_kv_cpu]
    
    # Enable query attention
    enable_attention_prefill_query(args.model, model_llama, args.temperature, args.scale)
    
    # Generate final output
    input_ids = torch.cat([prefix_input_ids, context_input_ids, query_input_ids], dim=-1)
    outputs = model_llama.generate(
        input_ids=input_ids,
        past_key_values=context_kv_gpu,
        max_new_tokens=512,
        temperature=args.temperature
    )

if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)
    generate(args)
