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
from typing import List
import torch
from logits_processor_zoo.utils import text_to_token, get_new_line_tokens


class MultipleChoiceLogitsProcessor:
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
        self.very_large_number = 999

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        for row_ind in range(input_ids.shape[0]):
            if self.boost_first_words:
                choice = 0

                first_tokens = []
                for i in range(len(input_ids[row_ind]) - 3):
                    # A choice is like "\nA) hair dryer", where first token is "hair"
                    choice_starts = (
                            (input_ids[row_ind, i].item() in self.new_line_tokens) and
                            (input_ids[row_ind, i + 1] == self.choice_tokens[choice]) and
                            (input_ids[row_ind, i + 2] == self.delimiter_token)
                    )

                    if choice_starts:
                        first_tokens.append(input_ids[row_ind, i + 3])
                        choice += 1

                        if choice >= len(self.choice_tokens):
                            break

                boost = self.boost_first_words * scores[row_ind, first_tokens]
                scores[row_ind, self.choice_tokens[:len(first_tokens)]] += boost

        scores[:, self.choice_tokens] += self.very_large_number

        return scores
