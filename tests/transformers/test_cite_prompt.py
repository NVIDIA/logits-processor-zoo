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

from logits_processor_zoo.transformers import CiteFromPromptLogitsProcessor


def test_cite_from_prompt_logits_processor(llm_runner):
    example_prompts = [
        "Please describe what macaques are.",
        "Tell me a story about a kid lost in forest."
    ]

    default_gen_output = llm_runner.generate_response(example_prompts, max_new_tokens=10)

    logits_processors = [CiteFromPromptLogitsProcessor(llm_runner.tokenizer, boost_factor=50.0,
                                                       conditional_boost_factor=50.0)]
    processed_gen_output = llm_runner.generate_response(example_prompts, logits_processors,  max_new_tokens=10)

    for prompt, default_out, processed_out in zip(example_prompts, default_gen_output, processed_gen_output):
        prompt_tokens = set(prompt.split())
        default_out_tokens = set(default_out.split())
        processed_out_tokens = set(processed_out.split())

        default_shared_tokens = prompt_tokens.intersection(default_out_tokens)
        processed_shared_tokens = prompt_tokens.intersection(processed_out_tokens)

        assert len(processed_shared_tokens) > len(default_shared_tokens)
