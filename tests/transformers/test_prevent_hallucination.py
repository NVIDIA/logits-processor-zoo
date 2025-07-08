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

from logits_processor_zoo.transformers import PreventHallucinationLogitsProcessor


def test_gen_length_logits_processor(llm_runner):
    example_prompts = ["Please describe what macaques are.", "Tell me a story about a kid lost in forest."]

    logits_processors = [PreventHallucinationLogitsProcessor(llm_runner.tokenizer, batch_size=2, minp=0.99, tolerate=2)]
    processed_gen_output = llm_runner.generate_response(example_prompts, logits_processors)

    assert all(["I don't know" in out for out in processed_gen_output])
