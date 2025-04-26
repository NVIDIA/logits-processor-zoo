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
from logits_processor_zoo.utils import text_to_token


class GenLengthLogitsProcessor:
    """
    A logits processor that adjusts the likelihood of the end-of-sequence (EOS) token
    based on the length of the generated sequence, encouraging or discouraging shorter answers.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    boost_factor (float): A factor to boost the likelihood of the EOS token as the sequence length increases.
                        Suggested value range is [-1.0, 1.0]. Negative values are used for the opposite effect.
    p (int, optional): The power to which the token count is raised when computing the boost value. Default is 2.
    complete_sentences (bool, optional): If True, boosts EOS token likelihood only when the last token is a full stop
                                        or a new line. Default is False.
    boost_token_str (str, optional): A string to be tokenized and used instead of EOS. Especially useful for </think>.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, boost_factor: float,
                 p: int = 2, complete_sentences: bool = False, boost_token_str: str = None):
        self.boost_token = tokenizer.eos_token_id
        if boost_token_str is not None:
            self.boost_token = text_to_token(tokenizer, boost_token_str, last=False)
        self.boost_factor = boost_factor
        self.p = p
        self.full_stop_token = text_to_token(tokenizer, "It is a sentence.", last=True)
        self.new_line_token = text_to_token(tokenizer, "It is a new line\n", last=True)
        self.complete_sentences = complete_sentences
        self.tokenizer = tokenizer

    def clone(self):
        return GenLengthLogitsProcessor(self.tokenizer, self.boost_factor, self.p,
                                        self.complete_sentences, self.boost_token_str)

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        gen_length = len(past_token_ids)

        boost_val = 0
        if not (self.boost_token in past_token_ids):
            boost_val = self.boost_factor * (gen_length ** self.p) / (10 ** self.p)

        if self.complete_sentences and gen_length > 0:
            enabled = (past_token_ids[-1] == self.full_stop_token) | (past_token_ids[-1] == self.new_line_token)
            scores[self.boost_token] += enabled * boost_val
        else:
            scores[self.boost_token] += boost_val

        return scores
