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

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList


class LLMRunner:
    def __init__(self, model_name='google/gemma-1.1-2b-it'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response(self, prompts, logits_processor_list=None, max_new_tokens=1000):
        if logits_processor_list is None:
            logits_processor_list = []

        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True)["input_ids"]

        out_ids = self.model.generate(input_ids.to(self.model.device),
                                      max_new_tokens=max_new_tokens, min_new_tokens=1,
                                      logits_processor=LogitsProcessorList(logits_processor_list)
                                      )

        gen_output = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)

        return [out[len(prompt):].strip() for prompt, out in zip(prompts, gen_output)]


@pytest.fixture(scope='session')
def llm_runner():
    return LLMRunner(model_name="MaxJeblick/llama2-0b-unit-test")
