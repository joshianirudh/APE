import torch
import math

from typing import Dict, List, Literal, Optional, Tuple, Union

from transformers.models.gemma2.modeling_gemma2 import (
    rotate_half,
    repeat_kv,
    Gemma2Attention,
    Gemma2Model
)

from torch import nn
import types

from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from flash_attn import flash_attn_func, flash_attn_varlen_func

from transformers.modeling_flash_attention_utils import _upad_input

from transformers.modeling_outputs import BaseModelOutputWithPast

def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def gemma_parallel_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states_raw = self.q_proj(hidden_states)
    key_states_raw = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states_raw = query_states_raw.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states_raw = key_states_raw.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if self.stage == -1:
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states_raw, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states_raw, cos, sin, position_ids)

        past_key_value = (key_states_raw, value_states, position_ids) if use_cache else None

        self.stage = 0
        self.len_instruction = q_len
    elif self.stage == 0 and (past_key_value is None or len(past_key_value) == 3):
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
        
        self.stage = 0

    else:
        assert past_key_value is not None

        if len(past_key_value) == 4:
            past_key, past_value, past_position = past_key_value[0], past_key_value[1], past_key_value[2]
            current_position = past_position.max().item() + 1
            self.len_document = past_key.shape[2] - self.len_instruction
            self.temperature = None
        else:
            (past_key, past_value, past_position) = past_key_value
            current_position = past_position.max().item() + 1


        key_states_raw = torch.cat([past_key, key_states_raw], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)
        key_position_ids = position_ids - position_ids.min().item() + current_position
        position_states = torch.cat([past_position, key_position_ids], dim=-1)
    
        past_key_value = (key_states_raw, value_states, position_states) if use_cache else None

        cos, sin = self.rotary_emb(value_states, position_states)
        query_states = apply_rotary_pos_emb_single(query_states_raw, cos[:,-key_position_ids.shape[1]:,:], sin[:,-key_position_ids.shape[1]:,:], key_position_ids)
        key_states = apply_rotary_pos_emb_single(key_states_raw, cos, sin, position_states)
        
        self.stage = 2

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    temperature = 0.6
    if self.stage == 2:  
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        causal_mask = torch.full(
            (attn_weights.shape[2], attn_weights.shape[3]), fill_value=torch.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype, device=attn_weights.device
        )
        if attn_weights.shape[2] != 1:
            causal_mask = torch.triu(causal_mask, diagonal=attn_weights.shape[3]-attn_weights.shape[2] + 1)
            attn_weights = attn_weights + causal_mask.unsqueeze(dim=0).unsqueeze(dim=0)

        attn_weights_inst = torch.logsumexp(attn_weights[:, :, :, :self.len_instruction], dim=-1)
        attn_weights_inst_norm = nn.functional.softmax(attn_weights[:, :, :, :self.len_instruction], dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_inst = torch.matmul(attn_weights_inst_norm, value_states[:, :, :self.len_instruction, :])

        attn_weights_doc = torch.logsumexp(attn_weights[:, :, :, self.len_instruction:self.len_instruction+self.len_document] / temperature, dim=-1) * temperature * 1.0
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
        flash_kwargs = {}
        flash_kwargs["softcap"] = self.config.attn_logit_softcapping
        if q_len > 1:
            if attention_mask is None:
                attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), softmax_scale=self.scaling, causal=True, **flash_kwargs)
                attn_output = attn_output.transpose(1, 2).contiguous()
            else:
                causal_mask = attention_mask
                if attention_mask is not None:
                    causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False,
                    scale=self.scaling,
                )

        else:
            attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), softmax_scale=self.scaling, causal=False, **flash_kwargs)
            attn_output = attn_output.transpose(1, 2).contiguous()

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

    '''
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "sliding_window": self.sliding_window,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if self.config.attn_logit_softcapping is not None:
        attn_weights = attn_weights / self.config.attn_logit_softcapping
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.config.attn_logit_softcapping

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
    '''

def gemma_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
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
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    # embed positions
    hidden_states = inputs_embeds

    # normalized
    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
    hidden_states = hidden_states * normalizer

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = ()

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

def gemma_update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: List[torch.FloatTensor],
    output_attentions: bool,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if past_key_values is not None:
        target_length = past_key_values[0][0].shape[2] + sequence_length
    else:
        target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask

def enable_gemma_parallel_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gemma_parallel_attention(
                module,
            )

        if isinstance(module, Gemma2Attention):
            model._modules[name].forward = types.MethodType(
                gemma_parallel_attention_forward, model._modules[name]
            )

        if isinstance(module, Gemma2Model):
            model._modules[name].forward = types.MethodType(
                gemma_forward, model._modules[name]
            )

        if isinstance(module, Gemma2Model):
            model._modules[name]._update_causal_mask = types.MethodType(
                gemma_update_causal_mask, model._modules[name]
            )

def init_gemma_parallel_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            init_gemma_parallel_attention(
                module,
            )

        if isinstance(module, Gemma2Attention):
            model._modules[name].stage = -1