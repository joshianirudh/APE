import torch

from transformers.models.llama.modeling_llama import (
    rotate_half,
    repeat_kv,
    LlamaAttention,
    LlamaSdpaAttention,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb
)

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import math
import numpy as np
import types

from torch import nn

from flash_attn import flash_attn_func
from typing import Dict, List, Literal, Optional, Tuple, Union

from transformers.cache_utils import Cache

def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def gram_schmidt(v1, v2):
    """
    Applies Gram-Schmidt process to create an orthogonal vector to v1 in the plane of v1 and v2.
    """
    v2_orthogonal = v2 - torch.sum(v2 * v1, dim=-1, keepdim=True) * v1
    v2_norm = torch.norm(v2_orthogonal, dim=-1, keepdim=True)
    v2_orthogonal = torch.where(v2_norm > 1e-8, v2_orthogonal / v2_norm, torch.zeros_like(v2_orthogonal))
    return v2_orthogonal


def rotate_high_dim_tensor(A, B):
    """
    Rotate high-dimensional tensor A around the plane defined by A and B by a specified angle.
    
    Args:
    A (torch.Tensor): Tensor to rotate, shape (batch_size, dim) or (dim,)
    B (torch.Tensor): Axis of rotation, shape (batch_size, dim) or (dim,)
    angle (float or torch.Tensor): Angle(s) of rotation
    degrees (bool): If True, angle is in degrees; if False, in radians
    
    Returns:
    torch.Tensor: Rotated tensor(s)
    """

    # Ensure A and B have the same shape
    assert A.shape == B.shape, "A and B must have the same shape"
    batch_size, dim = A.shape
    
    # Normalize B
    B = B / torch.norm(B, dim=1, keepdim=True)

    angle = 120 * torch.pi / 180
    angle = torch.as_tensor(angle, dtype=A.dtype, device=A.device)
    
    # Ensure angle is a batch
    angle = angle.view(1).expand(batch_size)
    
    # Compute the component of A parallel to B
    A_parallel = torch.sum(A * B, dim=1, keepdim=True) * B
    
    # Compute the component of A perpendicular to B
    A_perp = A - A_parallel
    
    # Normalize A_perp
    A_perp_norm = torch.norm(A_perp, dim=1, keepdim=True)
    A_perp = torch.where(A_perp_norm > 1e-8, A_perp / A_perp_norm, torch.zeros_like(A_perp))
    
    # Compute the vector perpendicular to both A_perp and B
    C = gram_schmidt(A_perp, B)
    
    # Compute rotation in the plane defined by A_perp and C
    cos_angle = torch.cos(angle).unsqueeze(1)
    sin_angle = torch.sin(angle).unsqueeze(1)
    #print(cos_angle, sin_angle)
    
    rotated_perp = cos_angle * A_perp + sin_angle * C
    
    # Combine rotated perpendicular component with parallel component
    rotated_A = A_parallel + A_perp_norm * rotated_perp
    
    return rotated_A


def rotate_high_dim_tensor2(A):
    """
    Rotate high-dimensional tensor A around a randomly generated plane by a specified angle.
    The random plane is consistent across the batch.
    
    Args:
    A (torch.Tensor): Tensor to rotate, shape (batch_size, dim)
    
    Returns:
    torch.Tensor: Rotated tensor(s)
    """
    
    # Ensure A is 2D
    assert A.dim() == 2, "Input tensor A must be 2D (batch_size, dim)"
    batch_size, dim = A.shape
    
    # Generate a random tensor B with the same second dimension as A, but only one in the first dimension
    B = torch.randn(1, dim, dtype=A.dtype, device=A.device)
    
    # Normalize B
    B = B / torch.norm(B, dim=1, keepdim=True)
    
    # Expand B to match the batch size of A
    B = B.expand(batch_size, -1)

    angle = -0 * torch.pi / 180
    angle = torch.as_tensor(angle, dtype=A.dtype, device=A.device)
    
    # Ensure angle is a batch
    angle = angle.view(1).expand(batch_size)
    
    # Compute the component of A parallel to B
    A_parallel = torch.sum(A * B, dim=1, keepdim=True) * B
    
    # Compute the component of A perpendicular to B
    A_perp = A - A_parallel
    
    # Normalize A_perp
    A_perp_norm = torch.norm(A_perp, dim=1, keepdim=True)
    A_perp = torch.where(A_perp_norm > 1e-8, A_perp / A_perp_norm, torch.zeros_like(A_perp))
    
    # Compute the vector perpendicular to both A_perp and B
    C = gram_schmidt(A_perp, B)
    
    # Compute rotation in the plane defined by A_perp and C
    cos_angle = torch.cos(angle).unsqueeze(1)
    sin_angle = torch.sin(angle).unsqueeze(1)
    
    rotated_perp = cos_angle * A_perp + sin_angle * C
    
    # Combine rotated perpendicular component with parallel component
    rotated_A = A_parallel + A_perp_norm * rotated_perp
    
    return rotated_A

def rotate_3d_tensor(A, B):
    """
    Rotate 3D tensor A around the plane defined by A and B by random angles for each item in the first dimension.
    
    Args:
    A (torch.Tensor): Tensor to rotate, shape (batch_size, num_vectors, dim)
    B (torch.Tensor): Axis of rotation, shape (batch_size, num_vectors, dim)
    
    Returns:
    torch.Tensor: Rotated tensor(s)
    """
    # Ensure A and B have the same shape
    assert A.shape == B.shape, "A and B must have the same shape"
    batch_size, num_vectors, dim = A.shape
    
    # Normalize B
    B = B / torch.norm(B, dim=-1, keepdim=True)

    # Generate random angles for each item in the first dimension
    min_angle = -120 * torch.pi / 180  # 30 degrees in radians
    max_angle = -120 * torch.pi / 180  # 60 degrees in radians
    angle = (max_angle - min_angle) * torch.rand(batch_size, 1, 1, device=A.device, dtype=A.dtype) + min_angle
    
    # Compute the component of A parallel to B
    A_parallel = torch.sum(A * B, dim=-1, keepdim=True) * B
    
    # Compute the component of A perpendicular to B
    A_perp = A - A_parallel
    
    # Normalize A_perp
    A_perp_norm = torch.norm(A_perp, dim=-1, keepdim=True)
    A_perp = torch.where(A_perp_norm > 1e-8, A_perp / A_perp_norm, torch.zeros_like(A_perp))
    
    # Compute the vector perpendicular to both A_perp and B
    C = gram_schmidt(A_perp, B)
    
    # Compute rotation in the plane defined by A_perp and C
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    rotated_perp = cos_angle * A_perp + sin_angle * C
    
    # Combine rotated perpendicular component with parallel component
    rotated_A = A_parallel + A_perp_norm * rotated_perp
    
    return rotated_A

def rotate_3d_tensor2(A):
    """
    Rotate 3D tensor A around a randomly generated axis B by random angles for each item in the first dimension.
    
    Args:
    A (torch.Tensor): Tensor to rotate, shape (batch_size, num_vectors, dim)
    
    Returns:
    torch.Tensor: Rotated tensor(s)
    """
    batch_size, num_vectors, dim = A.shape

    torch.manual_seed(0)
    
    # Randomly generate B for each item in the batch
    B = torch.randn(batch_size, 1, dim, device=A.device, dtype=A.dtype)
    
    # Repeat B along the second dimension to match A's shape
    B = B.repeat(1, num_vectors, 1)
    
    # Normalize B
    B = B / torch.norm(B, dim=-1, keepdim=True)

    # Generate random angles for each item in the first dimension
    min_angle = -180 * torch.pi / 180  # 30 degrees in radians
    max_angle = -180 * torch.pi / 180  # 60 degrees in radians
    angle = (max_angle - min_angle) * torch.rand(batch_size, 1, 1, device=A.device, dtype=A.dtype) + min_angle
    
    # Compute the component of A parallel to B
    A_parallel = torch.sum(A * B, dim=-1, keepdim=True) * B
    
    # Compute the component of A perpendicular to B
    A_perp = A - A_parallel
    
    # Normalize A_perp
    A_perp_norm = torch.norm(A_perp, dim=-1, keepdim=True)
    A_perp = torch.where(A_perp_norm > 1e-8, A_perp / A_perp_norm, torch.zeros_like(A_perp))
    
    # Compute the vector perpendicular to both A_perp and B
    C = gram_schmidt(A_perp, B)
    
    # Compute rotation in the plane defined by A_perp and C
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    rotated_perp = cos_angle * A_perp + sin_angle * C
    
    # Combine rotated perpendicular component with parallel component
    rotated_A = A_parallel + A_perp_norm * rotated_perp
    
    return rotated_A


def rotate_and_scale_3d_tensor(A, B, scale_range=(0.5, 2)):
    """
    Rotate 3D tensor A around the plane defined by A and B by random angles between 30 and 60 degrees 
    for each item in the first dimension, and apply a random scaling factor for each batch.
    
    Args:
    A (torch.Tensor): Tensor to rotate and scale, shape (batch_size, num_vectors, dim)
    B (torch.Tensor): Axis of rotation, shape (batch_size, num_vectors, dim)
    scale_range (tuple): Range for random scaling factor (min, max)
    
    Returns:
    torch.Tensor: Rotated and scaled tensor(s)
    """
    # Ensure A and B have the same shape
    assert A.shape == B.shape, "A and B must have the same shape"
    batch_size, num_vectors, dim = A.shape
    
    # Normalize B
    B = B / torch.norm(B, dim=-1, keepdim=True)

    # Generate random angles between 30 and 60 degrees (in radians) for each item in the first dimension
    min_angle = 0 * torch.pi / 180  # 30 degrees in radians
    max_angle = 60 * torch.pi / 180  # 60 degrees in radians
    angle = (max_angle - min_angle) * torch.rand(batch_size, 1, 1, device=A.device, dtype=A.dtype) + min_angle
    
    # Generate random scaling factors for each batch
    min_scale, max_scale = scale_range
    scale = (max_scale - min_scale) * torch.rand(batch_size, 1, 1, device=A.device, dtype=A.dtype) + min_scale
    
    # Compute the component of A parallel to B
    A_parallel = torch.sum(A * B, dim=-1, keepdim=True) * B
    
    # Compute the component of A perpendicular to B
    A_perp = A - A_parallel
    
    # Normalize A_perp
    A_perp_norm = torch.norm(A_perp, dim=-1, keepdim=True)
    A_perp = torch.where(A_perp_norm > 1e-8, A_perp / A_perp_norm, torch.zeros_like(A_perp))
    
    # Compute the vector perpendicular to both A_perp and B
    C = gram_schmidt(A_perp, B)
    
    # Compute rotation in the plane defined by A_perp and C
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    rotated_perp = cos_angle * A_perp + sin_angle * C
    
    # Combine rotated perpendicular component with parallel component and apply scaling
    rotated_and_scaled_A = scale * (A_parallel + A_perp_norm * rotated_perp)
    
    return rotated_and_scaled_A

def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

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

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, (query_states, key_states, value_states), past_key_value

def llama_flash_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)


    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

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

    if self.stage == -1: # Stage 0: input the instruction
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states_raw, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states_raw, cos, sin, position_ids)

        #past_key_value = (key_states_raw, value_states, position_ids) if use_cache else None
        past_key_value = (key_states, value_states, position_ids) if use_cache else None

        self.stage = 0
        self.len_instruction = q_len
        #self.stage = 1
        #self.len_instruction = 0
    elif self.stage == 0 and (past_key_value is None or len(past_key_value) == 3): # Stage 1: input the long context in parallel
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
        
        #k_0 = key_states[:, :, 0, :].reshape(bsz, -1, self.head_dim)
        #q_0 = query_states[:, :, 0, :].reshape(bsz, self.num_key_value_heads, self.num_key_value_groups, self.head_dim).mean(dim=-2).reshape(bsz, -1, self.head_dim)
        #k_0_new = rotate_3d_tensor2(k_0).reshape(bsz, self.num_key_value_heads, 1, self.head_dim)
        #key_states[:, :, :1, :] = k_0_new

        #v_0 = value_states[:, :, 0, :].reshape(bsz, -1, self.head_dim)
        #q_0 = query_states[:, :, 0, :].reshape(bsz, self.num_key_value_heads, self.num_key_value_groups, self.head_dim).mean(dim=-2).reshape(bsz, -1, self.head_dim)
        #v_0_new = rotate_3d_tensor(v_0, q_0).reshape(bsz, self.num_key_value_heads, 1, self.head_dim)
        #value_states[:, :, :1, :] = v_0_new

        #past_key_value = (key_states_raw, value_states, position_states) if use_cache else None
        past_key_value = (key_states, value_states, position_states) if use_cache else None
        
        self.stage = 0
    else:
        assert past_key_value is not None

        if len(past_key_value) == 4:
            #self.A = True
            past_key, past_value, past_position = past_key_value[0], past_key_value[1], past_key_value[2]
            current_position = past_position.max().item() + 1
            self.len_document = past_key.shape[2] - self.len_instruction
            #self.num_document = (past_position == 0).sum().item()
            self.num_documents = past_key_value[3]
        else:
            #self.A = Fasle
            (past_key, past_value, past_position) = past_key_value
            current_position = past_position.max().item() + 1

        value_states = torch.cat([past_value, value_states], dim=2)
        key_position_ids = position_ids - position_ids.min().item() + current_position
        position_states = torch.cat([past_position, key_position_ids], dim=-1)
    
        #past_key_value = (key_states_raw, value_states, position_states) if use_cache else None


        cos, sin = self.rotary_emb(value_states, position_states)
        query_states = apply_rotary_pos_emb_single(query_states_raw, cos[:,-key_position_ids.shape[1]:,:], sin[:,-key_position_ids.shape[1]:,:], key_position_ids)
        
        key_states = apply_rotary_pos_emb_single(key_states_raw, cos[:,-key_position_ids.shape[1]:,:], sin[:,-key_position_ids.shape[1]:,:], key_position_ids)
        key_states= torch.cat([past_key, key_states], dim=2)
        past_key_value = (key_states, value_states, position_states) if use_cache else None
        #key_states = apply_rotary_pos_emb_single(key_states_raw, cos, sin, position_states)
        
        self.stage = 2

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)


    if self.stage == 2: 
        temperature = 0.9
        scale = 1.0
        #elif self.num_documents <= 5:
        #    temperature = 0.8
        #    scale = 0.8
        #else:
        #    temperature = 0.9
        #    scale = 0.8
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

        attn_weights_doc = torch.logsumexp(attn_weights[:, :, :, self.len_instruction:self.len_instruction+self.len_document] / temperature, dim=-1) * temperature * scale
        attn_weights_doc_norm = nn.functional.softmax(attn_weights[:, :, :, self.len_instruction:self.len_instruction+self.len_document] / temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_doc = torch.matmul(attn_weights_doc_norm, value_states[:, :, self.len_instruction:self.len_instruction+self.len_document, :])
        #if self.A is True:
        #    print(self.layer_idx, attn_weights_doc_norm[:, :, :, 0].mean())
        attn_weights_query = torch.logsumexp(attn_weights[:, :, :, self.len_instruction+self.len_document:], dim=-1)
        attn_weights_query_norm = nn.functional.softmax(attn_weights[:, :, :, self.len_instruction+self.len_document:], dim=-1, dtype=torch.float32).to(query_states.dtype)
        value_query = torch.matmul(attn_weights_query_norm, value_states[:, :, self.len_instruction+self.len_document:, :])

        attn_weights = torch.cat([attn_weights_inst.unsqueeze(-1), attn_weights_doc.unsqueeze(-1), attn_weights_query.unsqueeze(-1)], dim=-1).unsqueeze(dim=-2)
        #attn_weights = torch.cat([attn_weights_doc.unsqueeze(-1), attn_weights_query.unsqueeze(-1)], dim=-1).unsqueeze(dim=-2)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        #if self.layer_idx == 20:
        #    print(attn_weights)
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

def llama_causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states[:, -1:, :])
    logits = logits.float()

    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def enable_llama_parallel_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_parallel_attention(
                module,
            )

        if isinstance(module, LlamaSdpaAttention):
            model._modules[name].forward = types.MethodType(
                llama_parallel_attention_forward, model._modules[name]
            )

        if isinstance(module, LlamaModel):
            model._modules[name].forward = types.MethodType(
                llama_forward, model._modules[name]
            )
    if isinstance(model, LlamaForCausalLM):
        model.forward = types.MethodType(
            llama_causal_forward, model
        )

def enable_llama_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_attention(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_attention_forward, model._modules[name]
            )

def init_llama_parallel_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            init_llama_parallel_attention(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].stage = -1

def enable_llama_flash_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_flash_attention(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_flash_attention_forward, model._modules[name]
            )