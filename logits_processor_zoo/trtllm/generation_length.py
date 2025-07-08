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
from transformers import PreTrainedTokenizer
import torch
from tensorrt_llm.sampling_params import LogitsProcessor
from logits_processor_zoo.utils import text_to_token, SentenceChecker


class GenLengthLogitsProcessor(LogitsProcessor, SentenceChecker):
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
    def __init__(self, tokenizer: PreTrainedTokenizer, boost_factor: float, p: int = 2,
                 complete_sentences: bool = False, boost_token_str: str = None):
        SentenceChecker.__init__(self, tokenizer)
        self.tokenizer = tokenizer
        self.boost_token = self.tokenizer.eos_token_id
        self.boost_token_str = boost_token_str
        if boost_token_str is not None:
            self.boost_token = text_to_token(self.tokenizer, boost_token_str, last=False)
        self.boost_factor = boost_factor
        self.p = p
        self.complete_sentences = complete_sentences
        self.token_count = 0

    def __call__(self, req_id: int, logits: torch.Tensor,
                 token_ids: List[List[int]], stream_ptr: Optional[int],
                 client_id: Optional[int]) -> None:

        boost_val = self.boost_factor * (self.token_count ** self.p) / (10 ** self.p)

        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            ids = torch.LongTensor(token_ids).to(logits.device, non_blocking=True)

            if self.complete_sentences:
                enabled = self._check_sentence_end(ids)
                logits[:, :, self.boost_token] += enabled * boost_val
            else:
                logits[:, :, self.boost_token] += boost_val

        self.token_count += 1
