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
import time
from transformers import PreTrainedTokenizer
import torch
from tensorrt_llm.sampling_params import LogitsProcessor
from logits_processor_zoo.utils import text_to_token, enforce_tokens, SentenceChecker


class MaxTimeLogitsProcessor(LogitsProcessor, SentenceChecker):
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
        SentenceChecker.__init__(self, tokenizer)
        self.tokenizer = tokenizer
        self.boost_token = self.tokenizer.eos_token_id
        self.boost_token_str = boost_token_str
        if boost_token_str is not None:
            self.boost_token = text_to_token(self.tokenizer, boost_token_str, last=False)
        self.complete_sentences = complete_sentences
        self.token_count = 0
        self.max_time = max_time
        self.start_time = time.time()

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:

        elapsed_time = time.time() - self.start_time
        time_exceeded = elapsed_time > self.max_time

        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            ids = torch.LongTensor(token_ids).to(logits.device, non_blocking=True)

            enabled = True
            if self.complete_sentences:
                enabled = self._check_sentence_end(ids)

            if time_exceeded and enabled:
                # enforce the EOS token
                for i in range(logits.shape[1]):
                    enforce_tokens(logits[0, i], [self.boost_token])

        self.token_count += 1
