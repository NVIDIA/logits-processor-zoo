#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List, Optional
import torch
from transformers import PreTrainedTokenizer
from tensorrt_llm.sampling_params import LogitsProcessor


class CiteFromPromptLogitsProcessor(LogitsProcessor):
    """
    A logits processor which boosts or diminishes the likelihood of tokens present in the prompt (and optionally
    EOS token) to encourage the model to generate tokens similar to those seen in the prompt or vice versa.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    boost_factor (float): A factor to boost the likelihood of the tokens from the prompt.
                            Negative values are used for the opposite effect.
    boost_eos (bool, optional): If True, boosts EOS token too.
    conditional_boost_factor (float, optional): A factor to boost the likelihood of the tokens based on previous token.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, boost_factor: float = 1.0, boost_eos: bool = True,
                 conditional_boost_factor: float = 0.0):
        self.tokenizer = tokenizer
        self.boost_factor = boost_factor
        self.eos_token_id = self.tokenizer.eos_token_id
        self.boost_eos = boost_eos
        self.conditional_boost_factor = conditional_boost_factor
        self.prompt_token_ids = None

    def _init_before_gen(self, token_ids):
        self.prompt_token_ids = list(token_ids[0])  # take first beam since all beams have the same prompt

    def __call__(self, req_id: int, logits: torch.Tensor,
                 token_ids: List[List[int]], stream_ptr: Optional[int],
                 client_id: Optional[int]) -> None:
        if self.prompt_token_ids is None:
            self._init_before_gen(token_ids)

        tokens = set(self.prompt_token_ids)
        if self.boost_eos:
            tokens.add(self.eos_token_id)

        tokens = [t for t in tokens if t < logits.shape[-1]]

        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            logits[:, :, tokens] += self.boost_factor

            if self.conditional_boost_factor != 0:

                for i in range(len(token_ids)):  # iterate over beams
                    tokens = set()
                    for prompt_token_idx in range(len(self.prompt_token_ids) - 1):
                        in_vocab = self.prompt_token_ids[prompt_token_idx + 1] < logits.shape[-1]
                        last_token = self.prompt_token_ids[prompt_token_idx] == token_ids[i][-1]
                        if last_token and in_vocab:
                            tokens.add(self.prompt_token_ids[prompt_token_idx + 1])
                    logits[:, i, list(tokens)] += self.conditional_boost_factor
