# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import torch
from transformers.models.umt5.modeling_umt5 import (
    UMT5EncoderModel,
    UMT5LayerNorm,
)


class QEffUMT5LayerNorm(UMT5LayerNorm):
    def forward(self, hidden_states):
        div_first = hidden_states * torch.rsqrt(torch.tensor(hidden_states.shape[-1], dtype=torch.float32))
        variance = div_first.pow(2).sum(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float()) * hidden_states


class QEffUMT5EncoderModel(UMT5EncoderModel):
    # only change is we are are returning  encoder_outputs.last_hidden_state instead of encoder_outputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs.last_hidden_state
