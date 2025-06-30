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

from typing import List, Optional
import torch
from transformers import PreTrainedTokenizer
from logits_processor_zoo.utils import enforce_tokens
from tensorrt_llm.sampling_params import LogitsProcessor


class PreventHallucinationLogitsProcessor(LogitsProcessor):
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
        self.tokenizer = tokenizer
        self.minp = minp
        self.tolerate = tolerate
        self.iterators = None
        self.minp_counts = None

    def _init_before_gen(self, beam_width):
        self.iterators = torch.zeros(beam_width, dtype=torch.int32)
        self.minp_counts = torch.zeros(beam_width, dtype=torch.int32)

    def __call__(self, req_id: int, logits: torch.Tensor,
                 token_ids: List[List[int]], stream_ptr: Optional[int],
                 client_id: Optional[int]) -> None:
        beam_width = len(token_ids)
        if self.iterators is None:
            self._init_before_gen(beam_width)

        beam_width = len(token_ids)
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            for i in range(beam_width):  # iterate over beams
                current_index = self.iterators[i].item()

                if logits[0, i, :].softmax(dim=-1).amax() < self.minp:
                    self.minp_counts[i] += 1

                if self.minp_counts[i] > self.tolerate and current_index == 0:
                    enforce_tokens(logits[0, i], [self.phrase_tokens[current_index]])
                    self.iterators[i] += 1
                elif len(self.phrase_tokens) > current_index > 0:
                    enforce_tokens(logits[0, i], [self.phrase_tokens[current_index]])
                    self.iterators[i] += 1
                elif current_index == len(self.phrase_tokens):
                    self.iterators[i] = 0
                    self.minp_counts[i] = 0
