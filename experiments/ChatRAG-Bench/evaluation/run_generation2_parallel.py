

from arguments2 import get_args
from dataset2 import load_data, get_inputs_parallel, get_inputs_parallel_retrival
import torch
import os
import json
import multiprocessing as mp
from typing import List, Any
from pathlib import Path
from functools import partial
from tqdm import tqdm
import numpy as np
import random

from transformers import AutoTokenizer, AutoModelForCausalLM

def get_prompt_list(args):

    ## get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    ## get input data
    if args.eval_dataset == "nq":
        query_path = os.path.join(args.data_folder, args.nq_query_path)
        context_path = os.path.join(args.data_folder, args.nq_context_path)
    elif args.eval_dataset == "arguana":
        query_path = os.path.join(args.data_folder, args.arguana_query_path)
        context_path = os.path.join(args.data_folder, args.arguana_context_path)
    elif args.eval_dataset == "fever":
        query_path = os.path.join(args.data_folder, args.fever_query_path)
        context_path = os.path.join(args.data_folder, args.fever_context_path)    
    elif args.eval_dataset == "fiqa":
        query_path = os.path.join(args.data_folder, args.fiqa_query_path)
        context_path = os.path.join(args.data_folder, args.fiqa_context_path)  
    elif args.eval_dataset == "msmacro":
        query_path = os.path.join(args.data_folder, args.msmacro_query_path)
        context_path = os.path.join(args.data_folder, args.msmacro_context_path)     
    elif args.eval_dataset == "quora":
        query_path = os.path.join(args.data_folder, args.quora_query_path)
        context_path = os.path.join(args.data_folder, args.quora_context_path)      
    elif args.eval_dataset == "scifact":
        query_path = os.path.join(args.data_folder, args.scifact_query_path)
        context_path = os.path.join(args.data_folder, args.scifact_context_path)     
    else:
        raise Exception("please input a correct eval_dataset name!")
    
    data_list = load_data(query_path, context_path)
    print("number of samples in the dataset:", len(data_list))
    return data_list
    #prompt_list = get_inputs_retrival(data_list, args.eval_dataset, tokenizer, num_ctx=args.num_ctx, max_output_len=args.out_seq_len)

    #return prompt_list

def write_to_separate_file(rank: int, results: List[str], output_folder: str, dataset_name: str):
    """Write results from each process to a separate file."""
    output_path = os.path.join(output_folder, f"{dataset_name}_output_rank_{rank}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')
    return output_path

def combine_output_files(output_folder: str, dataset_name: str, world_size: int):
    """Combine separate output files in order and delete them afterwards."""
    final_output = os.path.join(output_folder, f"{dataset_name}_output.txt")
    
    try:
        # Combine files
        with open(final_output, 'w', encoding='utf-8') as outfile:
            for rank in range(world_size):
                rank_file = os.path.join(output_folder, f"{dataset_name}_output_rank_{rank}.txt")
                if os.path.exists(rank_file):
                    try:
                        with open(rank_file, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    finally:
                        # Delete each rank file after reading
                        try:
                            os.remove(rank_file)
                            print(f"Deleted intermediate file: {rank_file}")
                        except Exception as e:
                            print(f"Warning: Could not delete file {rank_file}: {e}")
        
        print(f"Successfully combined results into {final_output}")
        return final_output
        
    except Exception as e:
        print(f"Error during file combination: {e}")
        # If there's an error, attempt to clean up any remaining rank files
        for rank in range(world_size):
            rank_file = os.path.join(output_folder, f"{dataset_name}_output_rank_{rank}.txt")
            if os.path.exists(rank_file):
                try:
                    os.remove(rank_file)
                except:
                    pass
        raise

def build_chat(prompt):
    prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt

def build_prefix(prompt):
    prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    return prompt

def build_suffix(prompt):
    prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt

def split_list_into_equal_parts(original_list, part_size):
    result = []
    list_size = len(original_list)
    
    # Calculate how many full parts we can have and how many items will be left over
    full_parts = list_size // part_size
    leftover = list_size % part_size
    
    current_index = 0
    
    # Add full parts of the specified size
    for _ in range(full_parts):
        result.append(original_list[current_index:current_index + part_size])
        current_index += part_size
    
    # Add the leftover part (if any)
    if leftover > 0:
        result.append(original_list[current_index:])
    
    return result

def pred(rank, world_size, query_list, context_list, max_length, tokenizer, model, bos_token, args):
    device = torch.device(f'cuda:{rank}')
    output_list = []
    prompt_list = get_inputs_parallel(query_list, context_list, args.eval_dataset, tokenizer, num_ctx=args.num_ctx, max_output_len=args.out_seq_len)
    from modeling_llama_parallel import enable_llama_parallel_attention
    enable_llama_parallel_attention(model)
    model = model.eval()
    model = model.to(device)
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        for (instruction, documents, query) in tqdm(prompt_list):
            instruction = build_prefix(instruction)
            query = build_suffix(query)
            special_tokens_kwargs = {
                    "add_special_tokens": False
                }
            instruction_input_ids = tokenizer(instruction, truncation=False, return_tensors="pt", **special_tokens_kwargs).input_ids
            query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", **special_tokens_kwargs).input_ids
            len_instruction = instruction_input_ids.shape[1]
            len_query = query_input_ids.shape[1]
            from modeling_llama_parallel import init_llama_parallel_attention
            init_llama_parallel_attention(model)
            past_key_values = None
            outputs = model(
                instruction_input_ids.to(model.device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values_instruct = outputs.past_key_values
            past_key_values = list(outputs.past_key_values)
            input_ids = instruction_input_ids
            #for document in documents:
            #    document_input_ids = tokenizer(document, return_tensors='pt', truncation=True, max_length=max_length).input_ids
            documents_lists = split_list_into_equal_parts(documents, 10)
            for documents in documents_lists:
                document_input_ids = tokenizer(documents, return_tensors='pt', padding=True, truncation=True, max_length=max_length - len_instruction - len_query - args.max_tokens, **special_tokens_kwargs).input_ids
                document_mask = (document_input_ids != tokenizer.pad_token_id).reshape(-1)
                document_mask_flip = (document_input_ids != tokenizer.pad_token_id).flip([1]).reshape(-1)
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
                    #past_key = torch.cat([past_key_value_global[0], 
                    #                      past_key_value_local[0].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                    #past_value = torch.cat([past_key_value_global[1], 
                    #                        past_key_value_local[1].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2) 
                    #past_position = torch.cat([past_key_value_global[2],
                    #                           past_key_value_local[2].repeat(bsz, 1).flatten()[document_mask].unsqueeze(0)], dim=1)
                    past_key_values_new.append((past_key, past_value, past_position))
                past_key_values = past_key_values_new
            past_key_values_new = []
            for past_key_value in past_key_values:
                past_key_values_new.append((past_key_value[0], past_key_value[1], past_key_value[2], len(documents)))
            del past_key_values
            del past_key_values_instruct
            del past_key_values_local
            del outputs
            '''
            for past_key_value, past_key_value_instruct in zip(outputs.past_key_values, past_key_values_instruct):
                bsz, num_heads, seq_len, _ = past_key_value[0].size()
                past_key = torch.cat([past_key_value_instruct[0], 
                                        past_key_value[0].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                past_value = torch.cat([past_key_value_instruct[1], 
                                        past_key_value[1].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
                past_position = torch.cat([past_key_value_instruct[2],
                                            past_key_value[2].repeat(bsz, 1).flatten()[document_mask].unsqueeze(0) + past_key_value_instruct[2].max().item() + 1], dim=1)
                past_key_values.append((past_key, past_value, past_position, temperature))
            '''
            input_ids = torch.cat([input_ids, query_input_ids], dim=-1)
            context_length = input_ids.shape[-1]
            generation_kwargs = {}
            #generation_kwargs["cache_implementation"] = None
            output = model.generate(
                input_ids=input_ids.to(model.device),
                max_new_tokens=args.max_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                past_key_values=past_key_values_new,
                **generation_kwargs
            )[0]
            generated_text = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            generated_text = generated_text.strip().replace("\n", " ")
            #print(generated_text)
            # print("generated_text:", generated_text)
            output_list.append(generated_text)

    write_to_separate_file(rank, output_list, args.output_folder, args.eval_dataset)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main():
    seed_everything(42)
    args = get_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    
    ## bos token for llama-3
    bos_token = "<|begin_of_text|>"

    ## get model_path
    #model_path = os.path.join(args.model_folder, args.model_name)
    model_path = args.model_id  

    ## get prompt_list
    query_list, context_list = get_prompt_list(args)

    ## get output_datapath
    output_datapath = os.path.join(args.output_folder, "%s_output.txt" % args.eval_dataset)

    ## run inference
    #sampling_params = SamplingParams(temperature=0, top_k=1, max_tokens=args.max_tokens)

    ## This changes the GPU support to 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    max_length = args.max_lengths

    #query_list = query_list[:10]
    chunk_size = (len(query_list) + world_size - 1) // world_size  # Calculate chunk size
    query_list_subsets = [query_list[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=pred, args=(rank, world_size, query_list_subsets[rank], context_list, max_length, \
                    tokenizer, model, bos_token, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    final_output = combine_output_files(
        args.output_folder,
        args.eval_dataset,
        world_size
    )
    print(f"Results combined into {final_output}")


if __name__ == "__main__":
    main()
