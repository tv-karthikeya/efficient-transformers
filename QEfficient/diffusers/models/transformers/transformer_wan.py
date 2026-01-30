# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QEfficient WAN Transformer Implementation

This module provides optimized implementations of WAN transformers
with various attention blocking strategies for memory efficiency and performance optimization.
The implementation includes multiple blocking modes: head-only, KV-blocking, Q-blocking,
and combined QKV-blocking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
import math
import os
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanTransformer3DModel,
    _get_qkv_projections,
)
from diffusers.utils import set_weights_and_activate_adapters

from QEfficient.diffusers.models.modeling_utils import (
    compute_blocked_attention,
    get_attention_blocking_config,
)


class QEffWanAttnProcessor(WanAttnProcessor):
    """
    QEfficient WAN Attention Processor with Memory-Efficient Blocking Strategies.

    This processor implements multiple attention blocking modes to reduce memory usage
    and enable processing of longer sequences. It supports:
    - Head blocking: Process attention heads in chunks
    - KV blocking: Process key-value pairs in blocks
    - Q blocking: Process query tokens in blocks
    - QKV blocking: Combined query, key, and value blocking

    Environment Variables:
        ATTENTION_BLOCKING_MODE: Controls blocking strategy ('kv', 'q', 'qkv', 'default')
        head_block_size: Number of attention heads to process per block
        num_kv_blocks: Number of blocks for key-value processing
        num_q_blocks: Number of blocks for query processing
    """

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Main attention processing pipeline with support for multiple blocking strategies.

        This method orchestrates the complete attention computation including:
        1. QKV projection and normalization
        2. Rotary position embedding application
        3. Attention computation with selected blocking strategy
        4. Output projection

        Args:
            attn (WanAttention): The attention module instance
            hidden_states (torch.Tensor): Input hidden states
            encoder_hidden_states (Optional[torch.Tensor]): Cross-attention encoder states
            attention_mask (Optional[torch.Tensor]): Attention mask
            rotary_emb (Optional[Tuple[torch.Tensor, torch.Tensor]]): Rotary embeddings (cos, sin)

        Returns:
            torch.Tensor: Processed hidden states after attention
        """
        # Project inputs to query, key, value
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # Apply layer normalization to queries and keys
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape for multi-head attention: (batch, seq, dim) -> (batch, seq, heads, head_dim)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply rotary position embeddings if provided
        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                """Apply rotary position embeddings to the input tensor."""
                # Split into real and imaginary parts for complex rotation
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2].type_as(hidden_states)
                sin = freqs_sin[..., 1::2].type_as(hidden_states)

                # Apply rotation: (x1 + ix2) * (cos + isin) = (x1*cos - x2*sin) + i(x1*sin + x2*cos)
                real = x1 * cos - x2 * sin
                img = x1 * sin + x2 * cos
                x_rot = torch.stack([real, img], dim=-1)
                return x_rot.flatten(-2).type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Get blocking configuration
        blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
        # Apply blocking using pipeline_utils
        hidden_states = compute_blocked_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            head_block_size,
            num_kv_blocks,
            num_q_blocks,
            blocking_mode=blocking_mode,
            attention_mask=attention_mask,
        )

        # Reshape back to original format
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        # Apply output projection layers
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class QEffWanAttention(WanAttention):
    """
    QEfficient WAN Attention module with optimized processor.

    This class extends the base WanAttention with QEfficient optimizations,
    automatically setting up the QEffWanAttnProcessor for memory-efficient
    attention computation.
    """

    def __qeff_init__(self):
        """Initialize the QEfficient attention processor."""
        processor = QEffWanAttnProcessor()
        self.processor = processor


class QEffWanTransformer3DModel(WanTransformer3DModel):
    """
    QEfficient 3D WAN Transformer Model with adapter support.

    This model extends the base WanTransformer3DModel with QEfficient optimizations.
    """

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        """
        Set the currently active adapters for use in the diffusion network.

        This method manages PEFT adapters, allowing for efficient fine-tuning
        and model customization without modifying the base model parameters.

        Args:
            adapter_names (Union[List[str], str]): Names of adapters to activate
            weights (Optional[Union[float, Dict, List[float], List[Dict], List[None]]]):
                Weights for each adapter. Can be:
                - Single float: Applied to all adapters
                - List of floats: One weight per adapter
                - Dict: Detailed weight configuration
                - None: Uses default weight of 1.0

        Raises:
            ValueError: If adapter names and weights lists have different lengths

        Note:
            - Adapters enable parameter-efficient fine-tuning
            - Multiple adapters can be active simultaneously with different weights
            - Weights control the influence of each adapter on the model output
        """
        # Normalize adapter names to list format
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # Expand weights into a list, one entry per adapter
        # Examples for 2 adapters: [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # Set None values to default of 1.0
        # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # Expand weights using model-specific scaling function
        # e.g. [{...}, 7] -> [{expanded dict...}, 7]
        scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[
            self.config._class_name
        ]  # updated to use WanTransformer3DModel
        weights = scale_expansion_fn(self, weights)
        set_weights_and_activate_adapters(self, adapter_names, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        temb: torch.Tensor,
        timestep_proj: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the 3D WAN Transformer.

        This method implements the complete forward pass including:
        1. Patch embedding of input
        2. Rotary embedding preparation
        3. Cross-attention with encoder states
        4. Transformer block processing
        5. Output normalization and projection

        Args:
            hidden_states (torch.Tensor): Input tensor to transform
            encoder_hidden_states (torch.Tensor): Cross-attention encoder states
            rotary_emb (torch.Tensor): Rotary position embeddings
            temb (torch.Tensor): Time embedding for diffusion process
            timestep_proj (torch.Tensor): Projected timestep embeddings
            encoder_hidden_states_image (Optional[torch.Tensor]): Image encoder states for I2V
            return_dict (bool): Whether to return a dictionary or tuple
            attention_kwargs (Optional[Dict[str, Any]]): Additional attention arguments

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]:
                Transformed hidden states, either as tensor or in a dictionary
        """
        # Prepare rotary embeddings by splitting along batch dimension
        rotary_emb = torch.split(rotary_emb, 1, dim=0)

        # Apply patch embedding and reshape for transformer processing
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Concatenate image and text encoder states if image conditioning is present
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Standard forward pass
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # Output normalization, projection & unpatchify
        if temb.ndim == 3:
            # Handle 3D time embeddings: batch_size, seq_len, inner_dim (WAN 2.2 T2V)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # Handle 2D time embeddings: batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Ensure tensors are on the same device as hidden_states
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Apply adaptive layer normalization with time conditioning
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

        # Final output projection
        hidden_states = self.proj_out(hidden_states)

        # Store output for return (compiler optimization)
        output = hidden_states

        # Return in requested format
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


############################### FFN blocking

class QEffWanFeedForward(FeedForward):

    def __qeff_init__(self):
        self.hidden_dim = self.net[0].proj.out_features # inner dim

        self.w1 = self.net[0].proj
        self.droput = self.net[1]
        self.w2 = self.net[2]


    def _forward_default(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward pass without blocking"""
        return self.w2(self.droput(torch.nn.functional.gelu(self.w1(x))))

    def forward_token_blocked(self, x: torch.Tensor) -> torch.Tensor:
        """
        Token blocking: Process tokens in blocks.
        Pattern: For each token_block:
            X = self.w1(token_block)
            Y = torch.nn.functional.gelu(X)
            W = self.droput(Y)
            output = self.w2(W)
        Result: concatenate along token dimension
        """
        _, seqlen, dim = x.shape
        token_block_size = int(os.environ.get('ffn_token_block_size', seqlen))

        if seqlen <= 512:
            return self._forward_default(x)

        num_token_blocks = math.ceil(seqlen / token_block_size)
        outputs = []

        for token_idx in range(num_token_blocks):
            start_idx = token_idx * token_block_size
            end_idx = min(start_idx + token_block_size, seqlen)
            token_block = x[:, start_idx:end_idx, :]

            X = self.w1(token_block)
            Y = torch.nn.functional.gelu(X)
            # Z = self.w3(token_block)
            # W = Y * Z
            W = self.droput(Y)

            output_block = self.w2(W)
            outputs.append(output_block)

        # Concatenate along token dimension
        return torch.cat(outputs, dim=1)  # [BS, seqlen, dim]

    def forward_weight_blocked(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weight blocking: Process hidden dimensions in chunks to reduce intermediate size.

        Pattern: For each hidden_block:
            X = gate_weights[hidden_block] @ input
            Y = nonlinearity(X)
            W = dropout(Y)
            result += down_weights[:, hidden_block] @ W
        """
        BS, seqlen, dim = x.shape
        weight_block_size = int(os.environ.get('ffn_weight_block_size', self.hidden_dim))

        if self.hidden_dim <= 2048:
            return self._forward_default(x)

        num_weight_blocks = math.ceil(self.hidden_dim / weight_block_size)
        result = torch.zeros(BS, seqlen, dim, device=x.device, dtype=x.dtype)

        for weight_idx in range(num_weight_blocks):
            h_start = weight_idx * weight_block_size
            h_end = min(h_start + weight_block_size, self.hidden_dim)

            # Extract weight blocks
            w1_block = self.w1.weight[h_start:h_end, :]
            # w3_block = self.w3.weight[h_start:h_end, :]
            w2_block = self.w2.weight[:, h_start:h_end]

            w1_bias = self.w1.bias[h_start:h_end] if self.w1.bias is not None else None
            # w3_bias = self.w3.bias[h_start:h_end] if self.w3.bias is not None else None
            w2_bias = self.w2.bias if (weight_idx == 0 and self.w2.bias is not None) else None

            X = F.linear(x, w1_block, w1_bias)
            Y = torch.nn.functional.gelu(X)
            # Z = F.linear(x, w3_block, w3_bias)
            # W = Y * Z
            W = self.droput(Y)

            result += F.linear(W, w2_block, w2_bias)

        return result

    def forward_token_weight_blocked(self, x: torch.Tensor) -> torch.Tensor:
        """
        Both token and weight blocking: Process tokens and hidden dims in chunks.

        Pattern: For each token_block:
                   For each hidden_block:
                     X = gate_weights[hidden_block] @ token_block
                     Y = nonlinearity(X)
                     W = dropout(Y)
                     result[token_positions] += down_weights[:, hidden_block] @ W
        """
        BS, seqlen, dim = x.shape
        token_block_size = int(os.environ.get('ffn_token_block_size', seqlen))
        weight_block_size = int(os.environ.get('ffn_weight_block_size', self.hidden_dim))

        if seqlen <= 512 and self.hidden_dim <= 2048:
            return self._forward_default(x)

        num_token_blocks = math.ceil(seqlen / token_block_size)  # seqlen = 5040
        num_weight_blocks = math.ceil(self.hidden_dim / weight_block_size) # self.hidden_dim = 13,824
        result = torch.zeros(BS, seqlen, dim, device=x.device, dtype=x.dtype)

        # token blocks
        for token_idx in range(num_token_blocks):
            t_start = token_idx * token_block_size
            t_end = min(t_start + token_block_size, seqlen)

            # Extract token block
            token_block = x[:, t_start:t_end, :]

            # weight blocks
            for weight_idx in range(num_weight_blocks):
                h_start = weight_idx * weight_block_size
                h_end = min(h_start + weight_block_size, self.hidden_dim)

                # Extract weight blocks
                w1_block = self.w1.weight[h_start:h_end, :]
                # w3_block = self.w3.weight[h_start:h_end, :]
                w2_block = self.w2.weight[:, h_start:h_end]

                w1_bias = self.w1.bias[h_start:h_end] if self.w1.bias is not None else None
                # w3_bias = self.w3.bias[h_start:h_end] if self.w3.bias is not None else None
                w2_bias = self.w2.bias if (weight_idx == 0 and self.w2.bias is not None) else None

                X = F.linear(token_block, w1_block, w1_bias)
                Y = torch.nn.functional.gelu(X)
                # Z = F.linear(token_block, w3_block, w3_bias)
                # W = Y * Z
                W = self.droput(Y)
                result[:, t_start:t_end, :] += F.linear(W, w2_block, w2_bias)

        return result

    def _get_ffn_blocking_mode(self):
        """Get FFN blocking mode from environment variable"""
        mode = os.environ.get('FFN_BLOCKING_MODE', 'default').lower()
        valid_modes = ['default', 'token', 'weight', 'token_weight']
        if mode not in valid_modes:
            raise ValueError(f"Invalid FFN_BLOCKING_MODE: {mode}. Must be one of {valid_modes}")
        return mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocking_mode = self._get_ffn_blocking_mode()
        if blocking_mode == "token":
            return self.forward_token_blocked(x)
        elif blocking_mode == "weight":
            return self.forward_weight_blocked(x)
        elif blocking_mode == "token_weight":
            return self.forward_token_weight_blocked(x)
        else:  # default
            return self._forward_default(x)
