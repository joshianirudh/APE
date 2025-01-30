import torch
import math

from typing import Dict, List, Literal, Optional, Tuple, Union

from transformers.models.mistral.modeling_mistral import (
    rotate_half,
    repeat_kv,
)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def forward(
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
    value_states_raw = self.v_proj(hidden_states)

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
    elif self.stage == 0:
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

    temperature = 0.9
    if self.stage == 2:  
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        causal_mask = torch.full(
            (attn_weights.shape[2], attn_weights.shape[3]), fill_value=torch.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype, device=attn_weights.device
        )
        if attn_weights.shape[2] != 1:
            causal_mask = torch.triu(causal_mask, diagonal=attn_weights.shape[3]-attn_weights.shape[2] + 1)
            attn_weights = attn_weights + causal_mask.unsqueeze(dim=0).unsqueeze(dim=0)

        attn_weights_inst = torch.logsumexp(attn_weights[:, :, :, :self.len_instruction], dim=-1)
        attn_weights_inst_norm = nn.functional.softmax(attn_weights[:, :, :, :self.len_instruction], dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_inst = torch.matmul(attn_weights_inst_norm, value_states[:, :, :self.len_instruction, :])

        attn_weights_doc = torch.logsumexp(attn_weights[:, :, :, self.len_instruction:self.len_instruction+self.len_document] / temperature, dim=-1) * temperature * 1
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

    #attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    #if attention_mask is not None:  # no matter the length, we just slice it
    #    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #    attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    #attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    #attn_output = torch.matmul(attn_weights, value_states)

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