
import json
import math
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def load_data(datapath):
    print("loading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    return data_list


def reformat_question(turn_list, dataset_name):

    ## only take the lastest 7 turns
    turn_list = turn_list[-7:]
    assert turn_list[-1]['role'] == 'user'

    long_answer_dataset_list = ["doc2dial", "quac", "qrecc", "inscit", "doqa_movies", "doqa_travel", "doqa_cooking", "hybridial", "convfinqa"]
    long_and_short_dataset_list = ["topiocqa"]
    entity_dataset_list = ["sqa"]
    short_dataset_list = ["coqa"]

    if dataset_name in long_answer_dataset_list:
        for item in turn_list:
            if item['role'] == 'user':
                ## only needs to add it on the first user turn
                item['content'] = 'Please give a full and complete answer for the question. ' + item['content']
                break
    
    elif dataset_name in long_and_short_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with a short span, or a full and complete answer. " + turn_list[-1]['content']

    elif dataset_name in entity_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with one or a list of items. " + turn_list[-1]['content']

    elif dataset_name in short_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with a short span. The answer needs to be just in a few words. " + turn_list[-1]['content']

    else:
        raise Exception("please input a correct dataset name!")
    
    question = ""
    for item in turn_list:
        if item["role"] == "user":
            question += "User: " + item["content"] + "\n\n"
        else:
            assert item["role"] == "assistant"
            question += "Assistant: " + item["content"] + "\n\n"
    
    question += "Assistant:"
    
    return question


def get_inputs(data_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=8192):

    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

    prompt_list = []
    for item in data_list:
        turn_list = item['messages']
        question_formatted = reformat_question(turn_list, dataset_name)

        ctx_list = [ctx["text"] for ctx in item['ctxs']]
        
        #tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
        #query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
        #context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')
        
        if len(item['ctxs']) > num_ctx:
            ctx_list = random.sample(ctx_list, num_ctx)

        #ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item['ctxs'][:num_ctx]]
        context = "".join(ctx_list)

        context_tokens = tokenizer.encode(context)
        question_tokens = tokenizer.encode(question_formatted)
        system_tokens = tokenizer.encode(system)

        print(len(context_tokens))
        if len(context_tokens) + len(question_tokens) + len(system_tokens) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[:max_seq_length - max_output_len - len(question_tokens) - len(system_tokens)]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        model_input = system + "\n\n" + context + "\n\n" + question_formatted

        prompt_list.append(model_input)
    
    return prompt_list

def get_inputs_retrival(data_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=8192):

    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

    #tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
    #query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder').cuda().eval()
    #context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder').cuda().eval()
    #tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
    #query_encoder = AutoModel.from_pretrained('thenlper/gte-base').cuda().eval()
    #context_encoder = AutoModel.from_pretrained('thenlper/gte-base').cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    query_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()
    context_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()
    
    prompt_list = []
    for item in tqdm(data_list):
        turn_list = item['messages']
        question_formatted = reformat_question(turn_list, dataset_name)

        ctx_list = [ctx["text"] for ctx in item['ctxs']]

       # def mean_pooling(token_embeddings, mask):
       #     token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
       #     sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
       #     return sentence_embeddings
    
        if len(item['ctxs']) > num_ctx:
            with torch.inference_mode():
                query_input = tokenizer(question_formatted, return_tensors='pt').input_ids.cuda()
                ctx_input = tokenizer(ctx_list, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.cuda()
                query_emb = query_encoder(query_input).last_hidden_state[:, 0, :]
                ctx_emb = context_encoder(ctx_input).last_hidden_state[:, 0, :]
                similarities = query_emb.matmul(ctx_emb.transpose(0, 1))
            indices = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)
            ctx_list = [ctx for i, ctx in enumerate(ctx_list) if i in indices[0][:num_ctx]]

        print(len(ctx_list))

        #ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item['ctxs'][:num_ctx]]
        context = "".join(ctx_list)

        context_tokens = tokenizer.encode(context)
        question_tokens = tokenizer.encode(question_formatted)
        system_tokens = tokenizer.encode(system)

        print(len(context_tokens))
        if len(context_tokens) + len(question_tokens) + len(system_tokens) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[:max_seq_length - max_output_len - len(question_tokens) - len(system_tokens)]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        model_input = system + "\n\n" + context + "\n\n" + question_formatted

        prompt_list.append(model_input)
    
    return prompt_list

def get_inputs_retrival2(data_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=8192):

    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

    tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
    query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder').cuda().eval()
    context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder').cuda().eval()
    #tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
    #query_encoder = AutoModel.from_pretrained('thenlper/gte-base').cuda().eval()
    #context_encoder = AutoModel.from_pretrained('thenlper/gte-base').cuda().eval()
    
    prompt_list = []
    for item in tqdm(data_list):
        turn_list = item['messages']
        question_formatted = reformat_question(turn_list, dataset_name)

        ctx_list = [ctx["text"] for ctx in item['ctxs']]

       # def mean_pooling(token_embeddings, mask):
       #     token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
       #     sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
       #     return sentence_embeddings
    
        if len(item['ctxs']) > num_ctx:
            with torch.inference_mode():
                query_input = tokenizer(question_formatted, return_tensors='pt').input_ids.cuda()
                ctx_input = tokenizer(ctx_list, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.cuda()
                query_emb = query_encoder(query_input).last_hidden_state[:, 0, :]
                ctx_emb = context_encoder(ctx_input).last_hidden_state[:, 0, :]
                similarities = query_emb.matmul(ctx_emb.transpose(0, 1))
            indices = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)
            ctx_list = [ctx for i, ctx in enumerate(ctx_list) if i in indices[0][:num_ctx]]

        print(len(ctx_list))

        #ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item['ctxs'][:num_ctx]]
        context = "".join(ctx_list)

        context_tokens = tokenizer.encode(context)
        question_tokens = tokenizer.encode(question_formatted)
        system_tokens = tokenizer.encode(system)

        print(len(context_tokens))
        if len(context_tokens) + len(question_tokens) + len(system_tokens) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[:max_seq_length - max_output_len - len(question_tokens) - len(system_tokens)]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        model_input = context + "\n\n" + system + "\n\n" + question_formatted

        prompt_list.append(model_input)
    
    return prompt_list


def get_inputs_parallel(data_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=4096):

    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

    prompt_list = []
    for item in data_list:
        turn_list = item['messages']
        question_formatted = reformat_question(turn_list, dataset_name)
        print(len(item['ctxs']))

        '''
        ctx_list_old = [ctx["text"] for ctx in item['ctxs']]
        ctx_list = []
        for i in range(0, len(ctx_list_old), 5):
            context = " ".join(ctx_list_old[i:i + 5])
            ctx_list.append(context)

        if len(ctx_list) > num_ctx:
            ctx_list = random.sample(ctx_list, num_ctx)
        '''
        
        ctx_list = [ctx["text"] for ctx in item['ctxs']]
        if len(item['ctxs']) > num_ctx:
            ctx_list = random.sample(ctx_list, num_ctx)

        contexts = [f"{ctx}" for ctx in ctx_list]
        
        print(len(contexts))

        model_input = (system + "\n\n", contexts, "\n\n" + question_formatted)

        prompt_list.append(model_input)
    
    return prompt_list

def get_inputs_parallel_retrival(data_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=4096):

    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

    prompt_list = []
    with torch.no_grad():
        #tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
        #query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder').cuda().eval()
        #context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder').cuda().eval()
        #tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
        #query_encoder = AutoModel.from_pretrained('thenlper/gte-base').cuda().eval()
        #context_encoder = AutoModel.from_pretrained('thenlper/gte-base').cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        query_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()
        context_encoder = AutoModel.from_pretrained('facebook/contriever').cuda().eval()
        for item in tqdm(data_list):
            turn_list = item['messages']
            question_formatted = reformat_question(turn_list, dataset_name)
            print(len(item['ctxs']))
            '''
            ctx_list_old = [ctx["text"] for ctx in item['ctxs']]
            ctx_list = []
            for i in range(0, len(ctx_list_old), 5):
                context = "".join(ctx_list_old[i:i + 5])
                ctx_list.append(context)
            '''
            ctx_list = [ctx["text"] for ctx in item['ctxs']]


            if len(ctx_list) > num_ctx:
                query_input = tokenizer(question_formatted, return_tensors='pt').input_ids.cuda()
                ctx_input = tokenizer(ctx_list, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.cuda()
                query_emb = query_encoder(query_input).last_hidden_state[:, 0, :]
                ctx_emb = context_encoder(ctx_input).last_hidden_state[:, 0, :]
                similarities = query_emb.matmul(ctx_emb.transpose(0, 1))
                indices = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)
                ctx_list = [ctx for i, ctx in enumerate(ctx_list) if i in indices[0][:num_ctx]]

            contexts = ctx_list
            
            print(len(contexts))
            
            '''
            contexts = []
            for context in ctx_list:
                context_tokens = tokenizer.encode(context)
                question_tokens = tokenizer.encode(question_formatted)
                system_tokens = tokenizer.encode(system)

                context_length = max_seq_length - (len(question_tokens) + len(system_tokens) + max_output_len)
                if len(context_tokens) > context_length:
                    n = math.ceil(len(context_tokens) / length)
                    length = (len_document + (n-1)) // n
                    insert_idx = (n+1) // 2 - 1
                    for i in range(n):
                        if i == insert_index:
                            group_length = (len_document + (n - 1)) // n - 1  # Make the middle group smaller
                        else:
                            group_length = (len_document + (n - 1)) // n  # Standard group size
                        contexts.append(tokenizer.decode(context_tokens[:group_length], skip_special_tokens=True))  # Take the first `group_length` elements
                        context_tokens = context_tokens[group_length:]
                else:
                    contexts.append(tokenizer.decode(context_tokens, skip_special_tokens=True))
            
            print(len(contexts))
            '''

            model_input = (system + "\n\n", contexts, "\n\n" + question_formatted)

            prompt_list.append(model_input)
    
    return prompt_list



