import json
import math
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def load_data(query_path, context_path):
    print("loading query from %s" % query_path)
    print("loading context from %s" % context_path)
    with open(query_path, "r") as f:
        query_list = [json.loads(line) for line in f]
    with open(context_path, "r") as f:
        context_list = [json.loads(line) for line in f]

    return query_list, context_list

def get_inputs(query_list, context_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=8192):

    system = "Find relevant documents and answer the question based on the documents."
    '''
    system = """
    You will be given a list of documents. You need to read carefully and understand all of them. Then you will be given a query, and your goal is to find all documents from the list that can help answer the query. Print out the ID and TITLE of each document.

    Your final answer should be a list of IDs, in the following format:
    Final Answer: [id1, id2, ...]
    If there is only one ID, it should be in the format:
    Final Answer: [id1]

    If there is no perfect answer output the closest one. Do not give an empty final answer.
    """
    '''

    prompt_list = []
    for item in query_list:
        query = item["query_text"]

        #ctx_list = [f'id: {ctx["pid"]}, title: {ctx["title_text"]}, passage: {ctx["passage_text"]}' for ctx in context_list]
        ctx_list = [ctx["passage_text"] for ctx in context_list]
        
        #tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
        #query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
        #context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')
        
        if len(ctx_list) > num_ctx:
            ctx_list = random.sample(ctx_list, num_ctx)

        #ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item['ctxs'][:num_ctx]]
        context = "".join(ctx_list)

        context_tokens = tokenizer.encode(context)
        question_tokens = tokenizer.encode(query)
        system_tokens = tokenizer.encode(system)

        print(len(context_tokens))
        if len(context_tokens) + len(question_tokens) + len(system_tokens) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[:max_seq_length - max_output_len - len(question_tokens) - len(system_tokens)]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        model_input = system + "\n\n" + context + "\n\n" + query

        prompt_list.append(model_input)
    
    return prompt_list

def get_inputs_retrival(query_list, context_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=8192):

    system = "Find relevant documents and answer the question based on the documents."
    '''
    system = """
    You will be given a list of documents. You need to read carefully and understand all of them. Then you will be given a query, and your goal is to find all documents from the list that can help answer the query. Print out the ID and TITLE of each document.

    Your final answer should be a list of IDs, in the following format:
    Final Answer: [id1, id2, ...]
    If there is only one ID, it should be in the format:
    Final Answer: [id1]

    If there is no perfect answer output the closest one. Do not give an empty final answer.
    """
    '''

    #tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
    #query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder').cuda().eval()
    #context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder').cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    query_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()
    context_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()

    prompt_list = []
    for item in query_list:
        query = item["query_text"]

        #ctx_list = [f'id: {ctx["pid"]}, title: {ctx["title_text"]}, passage: {ctx["passage_text"]}' for ctx in context_list]
        ctx_list = [ctx["passage_text"] for ctx in context_list]
        
        with torch.inference_mode():
            query_input = tokenizer(query, return_tensors='pt', truncation=True, max_length=512).input_ids.cuda()
            ctx_input = tokenizer(ctx_list, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.cuda()
            query_emb = query_encoder(query_input).last_hidden_state[:, 0, :]
            ctx_emb = context_encoder(ctx_input).last_hidden_state[:, 0, :]
            similarities = query_emb.matmul(ctx_emb.transpose(0, 1))
            indices = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)
            ctx_list = [ctx_list[i] for i in indices.squeeze()]

        print(len(ctx_list))

        #ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item['ctxs'][:num_ctx]]
        context = "".join(ctx_list)

        context_tokens = tokenizer.encode(context)
        question_tokens = tokenizer.encode(query)
        system_tokens = tokenizer.encode(system)

        print(len(context_tokens))
        if len(context_tokens) + len(question_tokens) + len(system_tokens) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[:max_seq_length - max_output_len - len(question_tokens) - len(system_tokens)]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        model_input = system + "\n\n" + context + "\n\n" + query

        prompt_list.append(model_input)
    
    return prompt_list

def get_inputs_parallel_retrival(query_list, context_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=4096):

    system = "Find relevant documents and answer the question based on the documents."

    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
    query_encoder = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5').cuda().eval()
    context_encoder = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5').cuda().eval()
    
    prompt_list = []
    for item in query_list:
        query = item["query_text"]

        #ctx_list = [f'id: {ctx["pid"]}, title: {ctx["title_text"]}, passage: {ctx["passage_text"]}' for ctx in context_list]
        ctx_list = [ctx["passage_text"] for ctx in context_list]
        
        with torch.inference_mode():
            query_input = tokenizer(query, return_tensors='pt', truncation=True, max_length=512).input_ids.cuda()
            ctx_input = tokenizer(ctx_list, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.cuda()
            query_emb = query_encoder(query_input).last_hidden_state[:, 0, :]
            ctx_emb = context_encoder(ctx_input).last_hidden_state[:, 0, :]
            similarities = query_emb.matmul(ctx_emb.transpose(0, 1))
            indices = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)
            ctx_list = [ctx_list[i] for i in indices.squeeze()]

        contexts = [f"{ctx}" for ctx in ctx_list]
        
        print(len(contexts))

        model_input = (system + "\n\n", contexts, "\n\n" + query)

        #model_input = ("", contexts, system + "\n\n" + query)

        prompt_list.append(model_input)
    
    return prompt_list


def get_inputs_parallel(query_list, context_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=4096):

    system = "Find relevant documents and answer the question based on the documents."
    
    prompt_list = []
    for item in query_list:
        query = item["query_text"]

        #ctx_list = [f'id: {ctx["pid"]}, title: {ctx["title_text"]}, passage: {ctx["passage_text"]}' for ctx in context_list]
        ctx_list = [ctx["passage_text"] for ctx in context_list]
        
        if len(ctx_list) > num_ctx:
            ctx_list = random.sample(ctx_list, num_ctx)

        contexts = [f"{ctx}" for ctx in ctx_list]
        
        print(len(contexts))

        model_input = (system + "\n\n", contexts, "\n\n" + query)

        #model_input = ("", contexts, system + "\n\n" + query)

        prompt_list.append(model_input)
    
    return prompt_list