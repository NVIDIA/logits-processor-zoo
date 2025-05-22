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

import torch
from transformers import PreTrainedTokenizer
from logits_processor_zoo.transformers.base import BaseLogitsProcessor


class CiteFromPromptLogitsProcessor(BaseLogitsProcessor):
    """
    A logits processor which boosts or diminishes the likelihood of tokens present in the prompt (and optionally
    EOS token) to encourage the model to generate tokens similar to those seen in the prompt or vice versa.
    WARNING: Create a new object before every model.generate call since every batch has different prompts.

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
        super().__init__()
        self.boost_factor = boost_factor
        self.eos_token_id = tokenizer.eos_token_id
        self.boost_eos = boost_eos
        self.conditional_boost_factor = conditional_boost_factor

    def _process(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        voc_size = scores.shape[1]
        for i in range(scores.shape[0]):
            tokens = set(self.prompt_token_ids[i])
            if self.boost_eos:
                tokens.add(self.eos_token_id)

            tokens = [t for t in tokens if t < voc_size]
            scores[i, tokens] += self.boost_factor

            if (self.conditional_boost_factor != 0) and (input_ids.shape[1] > self.prompt_token_ids.shape[1]):
                tokens = set()
                last_token = input_ids[i][-1]
                for j in range(len(self.prompt_token_ids[i]) - 1):
                    if (self.prompt_token_ids[i, j] == last_token) and (self.prompt_token_ids[i, j + 1] < voc_size):
                        tokens.add(self.prompt_token_ids[i, j + 1])
                scores[i, list(tokens)] += self.conditional_boost_factor

        return scores
