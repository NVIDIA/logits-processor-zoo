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
from logits_processor_zoo.utils import enforce_tokens


class PreventHallucinationLogitsProcessor:
    """
    A logits processor that mitigates hallucinated model outputs by enforcing a predefined fallback phrase
    when token confidence falls below a specified threshold.

    This processor monitors token probabilities during generation. If the model produces a number of
    low-confidence tokens (below `minp`) exceeding `tolerate`, it begins injecting a fallback phrase
    token-by-token to gracefully indicate uncertainty.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer used by the language model. It is used to tokenize the fallback phrase.
    minp : float, optional (default=0.4)
        The minimum probability threshold. Tokens with max probability below this are considered low-confidence.
    tolerate : int, optional (default=1)
        The number of consecutive low-confidence tokens tolerated before triggering the fallback phrase.
    phrase : str, optional (default="...I don't know actually.\\n")
        The phrase that will be inserted when hallucination is detected. It will be tokenized and injected
        sequentially into the generation.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, minp: float = 0.4, tolerate: int = 1,
                 phrase: str = "...I don't know actually.\n"):
        self.phrase = phrase
        self.eos_token_id = tokenizer.eos_token_id
        self.phrase_tokens = tokenizer.encode(self.phrase, add_special_tokens=False)
        self._reset()
        self.tokenizer = tokenizer
        self.minp = minp
        self.tolerate = tolerate

    def clone(self):
        return PreventHallucinationLogitsProcessor(self.tokenizer, self.minp, self.tolerate, self.phrase)

    def _reset(self):
        self.index = 0
        self.minp_count = 0

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        if not past_token_ids:  # new generation
            self._reset()

        if scores.softmax(dim=-1).amax() < self.minp:
            self.minp_count += 1

        if self.minp_count > self.tolerate and self.index == 0:
            scores = enforce_tokens(scores, [self.phrase_tokens[self.index]])
            self.index += 1
        elif len(self.phrase_tokens) > self.index > 0:
            scores = enforce_tokens(scores, [self.phrase_tokens[self.index]])
            self.index += 1
        elif self.index == len(self.phrase_tokens):
            self._reset()

        return scores
