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

from transformers import PreTrainedTokenizer
import torch
from logits_processor_zoo.utils import text_to_token
from logits_processor_zoo.transformers.base import BaseLogitsProcessor


class TriggerPhraseLogitsProcessor(BaseLogitsProcessor):
    """
    A logits processor which triggers phrases when it encounters given token.

    Parameters
    ----------
    phrase (str): The phrase to be generated by LLM when it encounters the trigger token.
    trigger_token_phrase (str): One token phrase in string to trigger phrases.
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    trigger_count (int): How many times the phrase will be triggered.
    trigger_after (bool): Whether the phrase is written after the trigger token or instead of the trigger token.
    """
    def __init__(self, phrase: str, trigger_token_phrase: str, tokenizer: PreTrainedTokenizer, batch_size: int,
                 trigger_count: int = 1, trigger_after: bool = False):
        super().__init__()
        self.trigger_token = text_to_token(tokenizer, trigger_token_phrase, last=False)
        self.phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)
        self.very_large_number = 999
        self.trigger_after = trigger_after
        self.batch_size = batch_size
        self.initial_trigger_count = trigger_count

    def _reset(self):
        self.iterators = -torch.ones(self.batch_size, dtype=torch.int32)
        self.trigger_count = self.initial_trigger_count*torch.ones(self.batch_size, dtype=torch.int32)

    def _process(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        for i in range(scores.shape[0]):
            if self.trigger_count[i] <= 0:
                continue

            it = self.iterators[i].item()
            if scores[i, :].argmax() == self.trigger_token and it == -1:
                self.iterators[i] = 0
                if not self.trigger_after:
                    scores[i, self.phrase_tokens[it]] = scores[i].max() + self.very_large_number
                    self.iterators[i] += 1
            elif len(self.phrase_tokens) > it >= 0:
                scores[i, self.phrase_tokens[it]] = scores[i].max() + self.very_large_number
                self.iterators[i] += 1

            if len(self.phrase_tokens) == self.iterators[i].item():  # phrase completed, reset for next trigger
                self.iterators[i] = -1
                self.trigger_count[i] -= 1

        return scores
