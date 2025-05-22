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

from typing import List
import torch
from transformers import PreTrainedTokenizer


class CiteFromPromptLogitsProcessor:
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
        self.eos_token_id = tokenizer.eos_token_id
        self.boost_eos = boost_eos
        self.conditional_boost_factor = conditional_boost_factor

    def clone(self):
        return CiteFromPromptLogitsProcessor(self.tokenizer, self.boost_factor, self.boost_eos,
                                             self.conditional_boost_factor)

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        tokens = set(prompt_tokens_ids)
        if self.boost_eos:
            tokens.add(self.eos_token_id)

        tokens = [t for t in tokens if t < scores.shape[0]]
        scores[tokens] += self.boost_factor

        if (self.conditional_boost_factor != 0) and (len(past_token_ids) > 0):
            tokens = set()
            last_token = past_token_ids[-1]
            for i in range(len(prompt_tokens_ids) - 1):
                if (prompt_tokens_ids[i] == last_token) and (prompt_tokens_ids[i + 1] < scores.shape[0]):
                    tokens.add(prompt_tokens_ids[i + 1])
            scores[list(tokens)] += self.conditional_boost_factor

        return scores
