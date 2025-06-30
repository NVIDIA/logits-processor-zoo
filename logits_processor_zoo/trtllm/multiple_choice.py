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
from typing import List, Optional
import torch
from logits_processor_zoo.utils import text_to_token, get_new_line_tokens, enforce_tokens
from tensorrt_llm.sampling_params import LogitsProcessor


class MultipleChoiceLogitsProcessor(LogitsProcessor):
    """
    A logits processor to answer multiple choice questions with one of the choices.
    A multiple choice question is like:
    I am getting a lot of calls during the day. What is more important for me to consider when I buy a new phone?
    0. Camera
    1. Screen resolution
    2. Operating System
    3. Battery
    The goal is to make LLM generate "3" as an answer.


    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    choices (List[str]): List of one character answers like A, B, C, D.
    delimiter (str): One character delimiter that comes after the choices like 1. or 2-.
    boost_first_words (float): Nonzero values add choices' first tokens' logits to boost performance.
                            Especially useful for the models which have difficulty associating the choice with its text.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, choices: List[str] = None,
                 delimiter: str = ".", boost_first_words: float = 0.0):
        if choices is None:
            choices = ["1", "2", "3", "4"]

        self.new_line_tokens = get_new_line_tokens(tokenizer)
        self.delimiter_token = text_to_token(tokenizer, delimiter, last=False)
        self.choice_tokens = [text_to_token(tokenizer, choice, last=False) for choice in choices]
        self.boost_first_words = boost_first_words
        self.first_tokens = list()

    def _init_choice_first_words(self, prompt_token_ids):
        choice = 0

        first_tokens = []
        for i in range(len(prompt_token_ids) - 3):
            # A choice is like "\nA) hair dryer", where first token is "hair"
            choice_starts = (
                    (prompt_token_ids[i] in self.new_line_tokens) and
                    (prompt_token_ids[i + 1] == self.choice_tokens[choice]) and
                    (prompt_token_ids[i + 2] == self.delimiter_token)
            )

            if choice_starts:
                first_tokens.append(prompt_token_ids[i + 3])
                choice += 1

                if choice >= len(self.choice_tokens):
                    break
        return first_tokens

    def __call__(self, req_id: int, logits: torch.Tensor,
                 token_ids: List[List[int]], stream_ptr: Optional[int],
                 client_id: Optional[int]) -> None:

        if len(self.first_tokens) == 0 and self.boost_first_words:
            prompt_token_ids = list(token_ids[0])  # take first beam since all beams have the same prompt
            self.first_tokens = self._init_choice_first_words(prompt_token_ids)

        beam_width = len(token_ids)
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            if len(self.first_tokens) > 0:
                boost = self.boost_first_words * logits[0, :, self.first_tokens]
                logits[0, :, self.choice_tokens[:len(self.first_tokens)]] += boost
            for i in range(beam_width):  # iterate over beams
                enforce_tokens(logits[0, i], self.choice_tokens)
