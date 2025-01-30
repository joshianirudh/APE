import copy
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from huggingface_hub import HfApi
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)


eval_logger = utils.eval_logger

from transformers.models.llama.modeling_llama import (
    rotate_half,
    repeat_kv,
    LlamaAttention,
    LlamaModel
)

from transformers.modeling_outputs import BaseModelOutputWithPast

import math
import numpy as np
import types

from torch import nn

from flash_attn import flash_attn_func

def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def llama_parallel_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states_raw = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states_raw = torch.cat(query_states_raw, dim=-1)

        key_states_raw = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states_raw = torch.cat(key_states_raw, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states_raw = self.q_proj(hidden_states)
        key_states_raw = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states_raw = query_states_raw.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states_raw = key_states_raw.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    
    #kv_seq_len = key_states_raw.shape[-2]
    #if past_key_value is not None:
    #    kv_seq_len += past_key_value[0].shape[-2]

    if self.stage == -1: # Stage 0: input the instruction
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states_raw, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states_raw, cos, sin, position_ids)

        past_key_value = (key_states_raw, value_states, position_ids) if use_cache else None

        self.stage = 0
        self.len_instruction = q_len
        #self.stage = 1
        #self.len_instruction = 0
    elif self.stage == 0 : # Stage 1: input the long context in parallel
        if past_key_value is not None:
            (past_key, past_value, past_position) = past_key_value
            key_states_raw = torch.cat([past_key, key_states_raw], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            position_states = torch.cat([past_position, position_ids], dim=-1)
            len_past = past_position.shape[1]
        else:
            position_states = position_ids
            len_past = 0

        cos, sin = self.rotary_emb(value_states, position_states)
        
        query_states = apply_rotary_pos_emb_single(query_states_raw, cos[:, len_past:], sin[:, len_past:], position_ids)
        key_states = apply_rotary_pos_emb_single(key_states_raw, cos, sin, position_states)

        past_key_value = (key_states_raw, value_states, position_states) if use_cache else None
        
        self.stage = 1
    else:
        assert past_key_value is not None

        if len(past_key_value) == 4:
            past_key, past_value, past_position = past_key_value[0], past_key_value[1], past_key_value[2]
            current_position = past_position.max().item() + 1
            self.len_document = past_key.shape[2] - self.len_instruction
            self.temperature = past_key_value[3].reshape(1, 1, 1, -1)
        else:
            (past_key, past_value, past_position) = past_key_value
            current_position = past_position.max().item() + 1
        #    if layer not in (0, 1):
        #        current_position = past_position.max().item() + 1
        #    else:
        #        current_position = past_position.max().item() + 1
        #    
        #    self.len_document = past_key.shape[2] - self.len_instruction
        #else:
        #    (past_key, past_value, past_position) = past_key_value
        #    current_position = past_position.max().item() + 1
        #current_position = past_position.max().item() + 1


        key_states_raw = torch.cat([past_key, key_states_raw], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)
        #print(past_position, position_ids)
        key_position_ids = position_ids - position_ids.min().item() + current_position
        #key_position_ids = torch.tensor(current_position, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0).unsqueeze(0)
        position_states = torch.cat([past_position, key_position_ids], dim=-1)
    
        past_key_value = (key_states_raw, value_states, position_states) if use_cache else None

        cos, sin = self.rotary_emb(value_states, position_states)
        query_states = apply_rotary_pos_emb_single(query_states_raw, cos[:,-key_position_ids.shape[1]:,:], sin[:,-key_position_ids.shape[1]:,:], key_position_ids)
        key_states = apply_rotary_pos_emb_single(key_states_raw, cos, sin, position_states)
        
        self.stage = 2

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    temperature = 0.8
    if self.stage == 2:  
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        '''
        a_doc = attn_weights[:, :, :, :self.len_document]
        v_doc = value_states[:, :, :self.len_document, :]
        assert sum(self.seq_len) == a_doc.size(-1)
        chunks_a = torch.split(a_doc, self.seq_len, dim=-1)
        chunks_v = torch.split(v_doc, self.seq_len, dim=-2)
        max_size = max(self.seq_len)
        min_size = min(self.seq_len)
        min_value = torch.finfo(a_doc.dtype).min

        # Pad each chunk on the last dimension and fill with min value of data type
        padded_chunks_a = []
        padded_chunks_v = []
        for (chunk_a, chunk_v) in zip(chunks_a, chunks_v):
            pad_size = max_size - chunk_a.size(-1)
            if pad_size > 0:
                # Pad with zeros first, then replace the padded values with the min value
                padded_chunk_a = F.pad(chunk_a, (0, pad_size))
                padded_chunk_a[:, :, :, -pad_size:] = min_value  # Fill the padded part with min value
                padded_chunk_v = F.pad(chunk_v, (0, 0, 0, pad_size))
                padded_chunk_v[:, :, -pad_size:, :] = min_value
            else:
                padded_chunk_a = chunk_a  
                padded_chunk_v = chunk_v
            
            padded_chunks_a.append(padded_chunk_a.unsqueeze(dim=3))
            padded_chunks_v.append(padded_chunk_v.unsqueeze(dim=2))
        chunks_a = torch.cat(padded_chunks_a, dim=3)
        chunks_v = torch.cat(padded_chunks_v, dim=2)
        _, indices = torch.topk(chunks_a, k=min_size, dim=-1, sorted=False)
        mask = torch.zeros_like(chunks_a, dtype=torch.bool).scatter(-1, indices, 1)
        chunks_a_topk = chunks_a[mask].reshape(1, self.num_heads, chunks_a.shape[2], -1)
        chunks_v_topk = chunks_v[mask.squeeze(dim=2)].reshape(1, self.num_heads, -1, self.head_dim)

        attn_weights_doc = torch.logsumexp(chunks_a_topk / temperature, dim=-1) * temperature
        attn_weights_doc_norm = nn.functional.softmax(chunks_a_topk / temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_doc = torch.matmul(attn_weights_doc_norm, chunks_v_topk)
        
        '''

        '''
        attn_weights_doc = torch.logsumexp(attn_weights[:, :, :, :self.len_document] / temperature, dim=-1) * temperature
        attn_weights_doc_norm = nn.functional.softmax(attn_weights[:, :, :, :self.len_document] / temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_doc = torch.matmul(attn_weights_doc_norm, value_states[:, :, :self.len_document, :])
  

        attn_weights_query = torch.logsumexp(attn_weights[:, :, :, self.len_document:], dim=-1)
        attn_weights_query_norm = nn.functional.softmax(attn_weights[:, :, :, self.len_document:], dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_query = torch.matmul(attn_weights_query_norm, value_states[:, :, self.len_document:, :])

        attn_weights = torch.cat([attn_weights_doc.unsqueeze(-1), attn_weights_query.unsqueeze(-1)], dim=-1)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        value_states = torch.cat([value_doc, value_query], dim=-2)

        attn_output = torch.matmul(attn_weights, value_states)
        '''

        causal_mask = torch.full(
            (attn_weights.shape[2], attn_weights.shape[3]), fill_value=torch.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype, device=attn_weights.device
        )
        if attn_weights.shape[2] != 1:
            causal_mask = torch.triu(causal_mask, diagonal=attn_weights.shape[3]-attn_weights.shape[2] + 1)
            attn_weights = attn_weights + causal_mask.unsqueeze(dim=0).unsqueeze(dim=0)

        attn_weights_inst = torch.logsumexp(attn_weights[:, :, :, :self.len_instruction], dim=-1)
        attn_weights_inst_norm = nn.functional.softmax(attn_weights[:, :, :, :self.len_instruction], dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_inst = torch.matmul(attn_weights_inst_norm, value_states[:, :, :self.len_instruction, :])

        attn_weights_doc = torch.logsumexp(attn_weights[:, :, :, self.len_instruction:self.len_instruction+self.len_document] / temperature, dim=-1) * temperature * 1.0 + math.log(0.7)
        attn_weights_doc_norm = nn.functional.softmax(attn_weights[:, :, :, self.len_instruction:self.len_instruction+self.len_document] / temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_doc = torch.matmul(attn_weights_doc_norm, value_states[:, :, self.len_instruction:self.len_instruction+self.len_document, :])
        
        attn_weights_query = torch.logsumexp(attn_weights[:, :, :, self.len_instruction+self.len_document:], dim=-1)
        attn_weights_query_norm = nn.functional.softmax(attn_weights[:, :, :, self.len_instruction+self.len_document:], dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_query = torch.matmul(attn_weights_query_norm, value_states[:, :, self.len_instruction+self.len_document:, :])

        attn_weights = torch.cat([attn_weights_inst.unsqueeze(-1), attn_weights_doc.unsqueeze(-1), attn_weights_query.unsqueeze(-1)], dim=-1).unsqueeze(dim=-2)
        #attn_weights = torch.cat([attn_weights_doc.unsqueeze(-1), attn_weights_query.unsqueeze(-1)], dim=-1).unsqueeze(dim=-2)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        value_states = torch.cat([value_inst.unsqueeze(-2), value_doc.unsqueeze(-2), value_query.unsqueeze(-2)], dim=-2)
        #value_states = torch.cat([value_doc.unsqueeze(-2), value_query.unsqueeze(-2)], dim=-2)

        attn_output = torch.matmul(attn_weights, value_states).squeeze(dim=-2)
    else:
        if q_len > 1:
            attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True)
        else:
            attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous()

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )


    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, None, output_attentions
    )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def enable_llama_parallel_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_parallel_attention(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_parallel_attention_forward, model._modules[name]
            )

        if isinstance(module, LlamaModel):
            model._modules[name].forward = types.MethodType(
                llama_forward, model._modules[name]
            )

def init_llama_parallel_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            init_llama_parallel_attention(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].stage = -1

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
    gpus: Optional[int] = None,
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu for device_idx in range(gpus)
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


@register_model("hf-auto", "hf", "huggingface")
class HFLM(TemplateLM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # optionally: take in an already-initialized transformers.PreTrainedModel
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self._model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()

            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = torch.device(device)

            # TODO: update this to be less of a hack once subfolder is fixed in HF
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        # determine which of 'causal' and 'seq2seq' backends to use
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
        )

        # if we passed `pretrained` as a string, initialize our model now
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                device_map_option=device_map_option,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                **kwargs,
            )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        if isinstance(pretrained, str) and (gpus >= 1 or str(self.device) == "mps"):
            # TODO: can remove this whole snippet except in the mps case, perhaps?
            if not (parallelize or autogptq or hasattr(self, "accelerator")):
                # place model onto device requested manually,
                # if not using HF Accelerate or device_map
                # or any other option that preloads model onto device
                try:
                    self.model.to(self.device)
                except ValueError:
                    eval_logger.debug(
                        "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                    )

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            if getattr(self.config, "model_type", None) == "qwen":
                # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
                self.tokenizer.pad_token = "<|endoftext|>"
            elif (
                self.tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
                or self.tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
            ):
                # The RWKV world tokenizer, does not allow for adding special tokens / setting the pad token (which is set as 0)
                # The additional tokenizer name check is needed, as there exists rwkv4 models with neox tokenizer
                # ---
                # Note that the world tokenizer class name, might change in the future for the final huggingface merge
                # https://github.com/huggingface/transformers/pull/26963
                assert self.tokenizer.pad_token_id == 0
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # TODO: override this for Gemma
        self.add_bos_token = add_bos_token
        if getattr(self.config, "model_type", None) == "gemma":
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if isinstance(pretrained, str):
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if parallelize:
                    if accelerator.num_processes > 1:
                        raise RuntimeError(
                            "Attempted to use both a HF Accelerate `device_map` and to launch via `accelerate launch`. If this is the case, please either remove `parallelize=True` from --model_args or launch outside of the Accelerate launcher."
                        )
                    else:
                        pass
                elif accelerator.num_processes == 1:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
                else:
                    if gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                    assert (
                        accelerator.distributed_type
                        in [
                            DistributedType.FSDP,
                            DistributedType.MULTI_GPU,
                            DistributedType.MULTI_NPU,
                        ]
                    ), "Unsupported distributed type provided. Only DDP and FSDP are supported."
                    if accelerator.distributed_type == DistributedType.FSDP:
                        self._model = accelerator.prepare(self.model)
                    else:
                        self._model = accelerator.prepare_model(
                            self.model, evaluation_mode=True
                        )
                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    if self.accelerator.is_local_main_process:
                        eval_logger.info(f"Using {gpus} devices with data parallelism")

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def chat_template(self) -> str:
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.chat_template
        return self.tokenizer.default_chat_template

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.
        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder))
        model type to be used.
        """
        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to AutoModelForCausalLM
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        assert self.AUTO_MODEL_CLASS in [
            transformers.AutoModelForCausalLM,
            transformers.AutoModelForSeq2SeqLM,
        ]
        return None

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        if parallelize:
            model_kwargs.update(
                _get_accelerate_args(
                    device_map_option,  # TODO: phase out device_map_option?
                    max_memory_per_gpu,
                    max_cpu_memory,
                    offload_folder,
                    gpus,
                )
            )
        elif "device_map" not in model_kwargs:
            # set a device_map to initialize model on the right GPU.
            # this is needed because it seems that the default behavior
            # for quantized models now seems to be device_map="auto"
            # which breaks data-parallel mode.
            if hasattr(self, "accelerator"):
                model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
            else:
                model_kwargs.update({"device_map": {"": str(self.device)}})

        if not autogptq:
            if model_kwargs.get("load_in_4bit", None):
                assert (
                    transformers.__version__ >= "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
            if transformers.__version__ >= "4.30.0":
                if model_kwargs.get("load_in_4bit", None):
                    if model_kwargs.get("bnb_4bit_compute_dtype", None):
                        model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(
                            model_kwargs["bnb_4bit_compute_dtype"]
                        )
            self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            #from lm_eval.models.modeling_llama_flash import LlamaForCausalContextLM
            #self._model = LlamaForCausalContextLM.from_pretrained(
            #    pretrained,
            #    revision=revision,
            #    torch_dtype=get_dtype(dtype),
            #    trust_remote_code=trust_remote_code,
            #    use_flash_attention_2="flash_attention_2",
            #    **model_kwargs,
            #)            
        else:
            try:
                from auto_gptq import AutoGPTQForCausalLM
            except ModuleNotFoundError:
                raise Exception(
                    "Tried to load auto_gptq, but auto-gptq is not installed ",
                    "please install auto-gptq via pip install lm-eval[gptq] or pip install -e .[gptq]",
                )

            self._model = AutoGPTQForCausalLM.from_quantized(
                pretrained,
                trust_remote_code=trust_remote_code,
                model_basename=None if autogptq is True else Path(autogptq).stem,
                use_safetensors=True
                if autogptq is True
                else autogptq.endswith(".safetensors"),
                **model_kwargs,
            )

        if peft and delta:
            raise ValueError(
                "Cannot use both 'peft' and 'delta' options at the same time."
            )

        if peft:
            if model_kwargs.get("load_in_4bit", None):
                if version.parse(PEFT_VERSION) < version.parse("0.4.0"):
                    raise AssertionError("load_in_4bit requires peft >= 0.4.0")
            if self._model.config.vocab_size != len(self.tokenizer):
                # resize model for LoRAs with added tokens
                self._model.resize_token_embeddings(len(self.tokenizer))
                eval_logger.info(
                    f"Model config indicates vocab_size='{self._model.config.vocab_size}', but found tokenizer with vocab size '{len(self.tokenizer)}'. Resizing model embedding layer..."
                )
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )
        elif delta:
            if autogptq:
                eval_logger.warning(
                    "Delta weights might trigger unexpected behavior when used with AutoGPTQ."
                )
            _model_delta = self.AUTO_MODEL_CLASS.from_pretrained(
                delta,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            for name, param in self._model.state_dict().items():
                try:
                    param.data += _model_delta.state_dict()[name]
                except KeyError:
                    raise KeyError(f"Delta model is missing weights for layer: {name}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to add delta weights to layer {name}. Error: {e}"
                    )

            del _model_delta

        enable_llama_parallel_attention(self._model)
        #from lm_eval.models.modeling_gemma_parallel import enable_gemma_parallel_attention
        #enable_gemma_parallel_attention(self._model)
        return None

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            #self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast_tokenizer,
            )
        return None

    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(self._model_call(test_batch, **call_kwargs), dim=-1)  # noqa: F841

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            clear_torch_cache()
            return batch_size

        clear_torch_cache()
        return batch_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}
        encoding = []
        encoding.append(self.tokenizer.encode(string[0] + "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", return_tensors="pt", **special_tokens_kwargs))
        encoding.append(self.tokenizer(string[1:-1], return_tensors="pt", padding=True, **special_tokens_kwargs))
        encoding.append(self.tokenizer.encode(string[-1], return_tensors="pt", **special_tokens_kwargs))
        #for example in string:
        #    encoding.append(self.tokenizer.encode(example, return_tensors="pt", **special_tokens_kwargs))
        #encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_encode2(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        #encoding = []
        #encoding.append(self.tokenizer.encode(string[0], return_tensors="pt", **special_tokens_kwargs))
        #encoding.append(self.tokenizer(string[1:-1], return_tensors="pt", padding=True, **special_tokens_kwargs))
        #encoding.append(self.tokenizer.encode(string[-1], return_tensors="pt", **special_tokens_kwargs))
        #for example in string:
        #    encoding.append(self.tokenizer.encode(example, return_tensors="pt", **special_tokens_kwargs))
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM

                init_llama_parallel_attention(self.model)
                instruction_input_ids = inps[0][0].to(self.model.device)
                len_instruction = len(instruction_input_ids[0])
                document_input_ids = inps[0][1].input_ids.to(self.model.device)
                document_mask = (document_input_ids != self.tokenizer.pad_token_id).reshape(-1)
                len_document = (document_input_ids != self.tokenizer.pad_token_id).sum(dim=1)
                temperature = torch.cat([torch.full((int(length),), len_document.sum()/(length * len_document.shape[0])) for length in len_document]).to(self.model.device)
                document_mask_flip = (document_input_ids != self.tokenizer.pad_token_id).flip([1]).reshape(-1)
                query_input_ids = inps[0][2].to(self.model.device)
                past_key_values = None
                outputs = self.model(
                    instruction_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values_instruct = outputs.past_key_values
                past_key_values = []
                for past_key_value in outputs.past_key_values:
                    bsz, _ = document_input_ids.shape
                    past_key = past_key_value[0].repeat(bsz, 1, 1, 1)
                    past_value = past_key_value[1].repeat(bsz, 1, 1, 1)
                    past_position = past_key_value[2]
                    past_key_values.append((past_key, past_value, past_position))
                outputs = self.model(
                    document_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = []

                for past_key_value in outputs.past_key_values:
                    bsz, num_heads, seq_len, _ = past_key_value[0].size()
                    past_key = torch.cat([past_key_value[0][:1, :, :len_instruction, :], 
                                        past_key_value[0][:, :, len_instruction:, :].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                    past_value = torch.cat([past_key_value[1][:1, :, :len_instruction, :], 
                                            past_key_value[1][:, :, len_instruction:, :].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
                    past_position = torch.cat([past_key_value[2][:, :len_instruction],
                                            past_key_value[2][:, len_instruction:].repeat(bsz, 1).flatten()[document_mask].unsqueeze(0)], dim=1)
                    past_key_values.append((past_key, past_value, past_position, temperature))
                '''
                for past_key_value, past_key_value_instruct in zip(outputs.past_key_values, past_key_values_instruct):
                    bsz, num_heads, seq_len, _ = past_key_value[0].size()
                    past_key = torch.cat([past_key_value_instruct[0], 
                                        past_key_value[0].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                    past_value = torch.cat([past_key_value_instruct[1], 
                                            past_key_value[1].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
                    past_position = torch.cat([past_key_value_instruct[2],
                                            past_key_value[2].repeat(bsz, 1).flatten()[document_mask].unsqueeze(0)], dim=1)
                    past_key_values.append((past_key, past_value, past_position, temperature))
                past_key_values = tuple(past_key_values)
                document_input_ids = document_input_ids.flatten()[document_mask].unsqueeze(0)
                '''
                len_token = query_input_ids.shape[-1]
                #pbar = tqdm(range(0, len_token - 1), leave=False)
                #pbar = range(0, len_token)
                #for idx in pbar:
                #    input_id = query_input_ids[:, idx: idx + 1]
                #    outputs = self.model(
                #        input_id,
                #        past_key_values=past_key_values,
                #        use_cache=True,
                #    )
                #    past_key_values = outputs.past_key_values
                outputs = self.model(
                    query_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                return outputs.logits
                #return self.model(inps).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        #stopping_criteria = stop_sequences_criteria(
        #    self.tokenizer, stop, context.shape[1], context.shape[0]
        #)
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 0, 1
        )

        set_seed(0)

        with torch.no_grad():
            #outputs = self.model.generate(
            #                    input_ids=context[2].input_ids.to(self.model.device), 
            #                    attention_mask=context[2].attention_mask.to(self.model.device),  
            #                    encoder_input_ids=context[1].input_ids.unsqueeze(0).to(self.model.device),
            #                    encoder_attention_mask=context[1].attention_mask.unsqueeze(0).to(self.model.device),
            #                    max_length=max_length + context[2].input_ids.shape[1])
            #return outputs
            init_llama_parallel_attention(self.model)
            #from lm_eval.models.modeling_gemma_parallel import init_gemma_parallel_attention
            #init_gemma_parallel_attention(self.model)
            instruction_input_ids = context[0].to(self.model.device)
            len_instruction = len(instruction_input_ids[0])
            document_input_ids = context[1].input_ids.to(self.model.device)
            document_mask = (document_input_ids != self.tokenizer.pad_token_id).reshape(-1)
            len_document = (document_input_ids != self.tokenizer.pad_token_id).sum(dim=1)
            temperature = torch.cat([torch.full((int(length),), len_document.sum()/(length * len_document.shape[0])) for length in len_document]).to(self.model.device)
            document_mask_flip = (document_input_ids != self.tokenizer.pad_token_id).flip([1]).reshape(-1)
            query_input_ids = context[2].to(self.model.device)
            past_key_values = None
            outputs = self.model(
                instruction_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values_instruct = outputs.past_key_values
            past_key_values = []
            for past_key_value in outputs.past_key_values:
                bsz, _ = document_input_ids.shape
                past_key = past_key_value[0].repeat(bsz, 1, 1, 1)
                past_value = past_key_value[1].repeat(bsz, 1, 1, 1)
                past_position = past_key_value[2]
                past_key_values.append((past_key, past_value, past_position))
            outputs = self.model(
                document_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = []

            for past_key_value in outputs.past_key_values:
                bsz, num_heads, seq_len, _ = past_key_value[0].size()
                past_key = torch.cat([past_key_value[0][:1, :, :len_instruction, :], 
                                      past_key_value[0][:, :, len_instruction:, :].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                past_value = torch.cat([past_key_value[1][:1, :, :len_instruction, :], 
                                        past_key_value[1][:, :, len_instruction:, :].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
                past_position = torch.cat([past_key_value[2][:, :len_instruction],
                                           past_key_value[2][:, len_instruction:].repeat(bsz, 1).flatten()[document_mask].unsqueeze(0)], dim=1)
                past_key_values.append((past_key, past_value, past_position, temperature))
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
            '''
            for past_key_value in outputs.past_key_values:
                bsz, num_heads, seq_len, _ = past_key_value[0].size()
                past_key = torch.cat([past_key_value[0].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                past_value = torch.cat([past_key_value[1].transpose(1, 2).flatten(0, 1)[document_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
                past_position = torch.cat([past_key_value[2].repeat(bsz, 1).flatten()[document_mask].unsqueeze(0)], dim=1)
                past_key_values.append((past_key, past_value, past_position, temperature))
            '''
            #past_key_values = tuple(past_key_values)
            document_input_ids = document_input_ids.flatten()[document_mask].unsqueeze(0)
            input_ids = torch.cat([instruction_input_ids, document_input_ids, query_input_ids], dim=-1)
            stopping_criteria = stop_sequences_criteria(
                self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
            )
            #generation_kwargs["cache_implementation"] = None
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length + instruction_input_ids.shape[1] + document_input_ids.shape[1] + query_input_ids.shape[1],
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                past_key_values=past_key_values,
                **generation_kwargs,
            )

            return outputs


    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            assert (
                contlen and inplen
            ), "Must pass input len and cont. len to select scored logits for causal LM"
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
            assert (
                contlen and not inplen
            ), "Selecting scored logits for Seq2SeqLM requires only cont. len"
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for (string,) in tqdm(
            [req.args for req in requests], disable=(disable_tqdm or (self.rank != 0))
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                requests=rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            #toks = req[1] + req[2]
            toks = req[1][2]
            return -len(toks), tuple(toks.squeeze().tolist())

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            #return req[-2] + req[-1][:-1]
            return req[1][2].squeeze().tolist()

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
            and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    #inp = torch.tensor(
                    #    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    #    dtype=torch.long,
                    #    device=self.device,
                    #)
                    #(inplen,) = inp.shape
                    cont = torch.tensor(
                        continuation_enc,
                        dtype=torch.long,
                        device=self.device,
                    ).unsqueeze(dim=0)
                    inp = [context_enc, cont[:, :-1]]
                    inplen = context_enc[-1].shape[1] + cont.shape[1] - 1
                elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                #batched_inps = pad_and_concat(
                #    padding_len_inp, inps, padding_side="right"
                #)  # [batch, padding_len_inp]
                batched_inps = inps[0]
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length
            context_enc = self.tok_encode(contexts[0])

            attn_masks = None

            if "max_length" not in kwargs:
                #kwargs["max_length"] = context_enc.shape[1] + max_gen_toks
                kwargs["max_length"] = max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    cont_toks = cont_toks[context_enc[0].shape[1] + (context_enc[1].input_ids != self.tokenizer.pad_token_id).sum() + context_enc[2].shape[1] :]
                    #cont_toks = cont_toks[context_enc[2].input_ids.shape[1] :]

                s = self.tok_decode(cont_toks)
                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]
                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str, revision: str) -> str:
            try:
                model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
                return model_info.sha
            except Exception as e:
                eval_logger.warn(
                    f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
                )
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        if self.peft:
            model_info["peft_sha"] = get_model_sha(self.peft, self.revision)
        if self.delta:
            model_info["delta_sha"] = get_model_sha(self.delta, self.revision)
        return model_info
