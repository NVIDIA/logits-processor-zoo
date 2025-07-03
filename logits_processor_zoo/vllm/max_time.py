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

import time
from typing import List
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from logits_processor_zoo.utils import text_to_token, enforce_tokens


class MaxTimeLogitProcessor:
    """
    A logits processor that increases the probability of generating the EOS token with time.
    The logit of the EOS token is boosted with an exponential function.
    After N seconds pass, the EOS token is forced.
    Optionally, after K < N seconds, we can trigger a sentence that forces the model to finish quickly.

    Parameters
    ----------
    max_time (float): Maximum time in seconds after which the EOS token must be forced.
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    boost_factor (float): A factor to boost the likelihood of the EOS token over time. Default is 1.0.
    complete_sentences (bool, optional): If True, boosts EOS token likelihood only when the last token is a full stop
                                        or a new line. Default is False.
    boost_token_str (str, optional): A string to be tokenized and used instead of EOS. Especially useful for </think>.

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_time: float,
        boost_factor: float,
        p: int = 2,
        complete_sentences: bool = False,
        boost_token_str: str = None,
    ):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.boost_token = self.tokenizer.eos_token_id
        self.boost_token_str = boost_token_str
        if boost_token_str is not None:
            self.boost_token = text_to_token(self.tokenizer, boost_token_str, last=False)
        self.boost_factor = boost_factor
        self.p = p
        self.full_stop_token = text_to_token(self.tokenizer, "It is a sentence.", last=True)
        self.new_line_token = text_to_token(self.tokenizer, "It is a new line\n", last=True)
        self.complete_sentences = complete_sentences
        self.max_time = max_time
        self._reset()

    # Mutable logits processor gets cloned for each prompt in a batch in order to prevent updating the same object
    # https://github.com/vllm-project/vllm/blob/19dcc02a72e3ed52e3bf95aae44ea1f40ce42ea0/vllm/sampling_params.py#L537-L550
    def clone(self):
        return MaxTimeLogitProcessor(
            self.tokenizer,
            self.max_time,
            self.boost_factor,
            self.p,
            self.complete_sentences,
            self.boost_token_str,
        )

    def _reset(self):
        self.start_time = time.time()

    def __call__(
        self,
        prompt_tokens_ids: List[int],
        past_token_ids: List[int],
        scores: torch.Tensor,
    ) -> torch.Tensor:

        elapsed_time = time.time() - self.start_time
        time_exceeded = elapsed_time > self.max_time
        gen_length = len(past_token_ids)

        boost_val = 0
        if not (self.boost_token in past_token_ids):
            boost_val = self.boost_factor * (elapsed_time**self.p) / (self.max_time**self.p)

        enabled = True
        if self.complete_sentences and gen_length > 0:
            enabled = (past_token_ids[-1] == self.full_stop_token) | (past_token_ids[-1] == self.new_line_token)

        if time_exceeded and enabled:
            scores = enforce_tokens(scores, [self.boost_token])
        else:
            scores[self.boost_token] += enabled * boost_val

        return scores
