

from arguments2 import get_args
from dataset2 import load_data, get_inputs, get_inputs_retrival
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

def pred(rank, world_size, query_list, context_list, max_length, tokenizer, model, bos_token, args):
    device = torch.device(f'cuda:{rank}')
    model = model.to(device).eval()
    prompt_list = get_inputs(query_list, context_list, args.eval_dataset, tokenizer, num_ctx=args.num_ctx, max_output_len=args.out_seq_len, max_seq_length=max_length)
    output_list = []
    #from modeling_llama_parallel import llama_causal_forward
    #model.forward = types.MethodType(
    #    llama_causal_forward, model
    #)
    for prompt in tqdm(prompt_list):
        prompt = build_chat(prompt)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        special_tokens_kwargs = {
                "add_special_tokens": False
            }
        input = tokenizer(prompt, truncation=False, return_tensors="pt", **special_tokens_kwargs).to(device)
        input_ids = input.input_ids
        context_length = input_ids.shape[-1]
        seq_len = input_ids.shape[-1]
        #while seq_len > 10000:
        #    with torch.no_grad():
        #        outputs = model(
        #            input_ids[:, :2000].to(model.device),
        #            past_key_values=past_key_values,
        #            use_cache=True,
        #        )
        #        past_key_values = outputs.past_key_values
        #        input_ids = input_ids[:, 10000:].to(model.device)
        #        seq_len = input_ids.size(1)
        with torch.inference_mode():
            past_key_values = None
            while seq_len > 10000:
                outputs = model(
                    input_ids[:, :10000].to(model.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                input_ids = input_ids[:, 10000:].to(model.device)
                seq_len = input_ids.size(1)
            output = model.generate(    
                **input,            
                max_new_tokens=args.max_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                past_key_values = past_key_values)
        generated_text = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
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
