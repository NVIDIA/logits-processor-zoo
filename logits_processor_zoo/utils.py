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


def text_to_token(tokenizer: PreTrainedTokenizer, text: str, last: bool):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if not last and len(tokens) > 2:
        # Usually the first token indicates the beginning, and the second token is our main token
        raise Exception(f"Can't convert {text} to token. It has {len(tokens)} tokens.")

    return tokens[-1]


def get_new_line_tokens(tokenizer: PreTrainedTokenizer):
    new_line_tokens = [token for token in tokenizer.get_vocab().values()
                       if tokenizer.decode(token).endswith("\n")]

    return set(new_line_tokens)


def enforce_tokens(scores: torch.Tensor, tokens: List[int]):
    choice_scores = scores[tokens].clone()
    gap = scores.max() - choice_scores.min()
    choice_scores += gap
    scores.fill_(scores.min())
    scores[tokens] = choice_scores
    return scores
