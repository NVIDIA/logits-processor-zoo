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
import torch
from transformers import PreTrainedTokenizer
from logits_processor_zoo.transformers.base import BaseLogitsProcessor
from logits_processor_zoo.utils import text_to_token, enforce_tokens, SentenceChecker


class MaxTimeLogitsProcessor(BaseLogitsProcessor, SentenceChecker):
    """
    A logits processor that enforces the end-of-sentence (EOS) token after a specified maximum time passes.
    Useful for controlling generation time and ensuring responses complete within time constraints.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    max_time (float): Maximum time (wall-clock time) in seconds after which the EOS token must be enforced.
    complete_sentences (bool, optional): If True, enforces EOS token only when the last token is a full stop
                                        or a new line. Default is False.
    boost_token_str (str, optional): A string to be tokenized and used instead of EOS.

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_time: float,
        complete_sentences: bool = False,
        boost_token_str: str = None,
    ):
        BaseLogitsProcessor.__init__(self)
        SentenceChecker.__init__(self, tokenizer)
        self.boost_token = tokenizer.eos_token_id
        if boost_token_str is not None:
            self.boost_token = text_to_token(tokenizer, boost_token_str, last=False)
        self.max_time = max_time
        self.complete_sentences = complete_sentences

    def _reset(self):
        self.start_time = time.time()

    def _process(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        elapsed_time = time.time() - self.start_time
        token_count = input_ids.shape[1] - self.prompt_token_ids.shape[1]

        enabled = (input_ids[:, -token_count:] == self.boost_token).sum(dim=1) == 0
        if self.complete_sentences:
            enabled = enabled & self._check_sentence_end(input_ids)

        if elapsed_time > self.max_time:
            for i in range(scores.shape[0]):
                if enabled[i]:
                    scores[i] = enforce_tokens(scores[i], [self.boost_token])
            return scores

        return scores
