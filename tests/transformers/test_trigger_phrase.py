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

from logits_processor_zoo.transformers import TriggerPhraseLogitsProcessor


def test_trigger_phrase_token_based_triggering(llm_runner):
    """Test that the phrase is triggered when the specified token is generated."""
    example_prompts = ["Query: "]

    trigger_token = "fig"
    phrase = "This is a triggered phrase."

    logits_processors = [
        TriggerPhraseLogitsProcessor(
            llm_runner.tokenizer,
            batch_size=len(example_prompts),
            phrase=phrase,
            trigger_token_phrase=trigger_token,
            trigger_after=True,
        )
    ]

    processed_gen_output = llm_runner.generate_response(example_prompts, logits_processors, max_new_tokens=1000)
    assert phrase in processed_gen_output[0]


def test_trigger_phrase_token_phrase_based_triggering(llm_runner):
    """Test that the phrase is triggered when the specified token is generated."""
    example_prompts = [
        "Generate a python function to calculate fibonacci numbers.",
        "Simple python function to calculate fibonacci numbers.",
    ]

    trigger_time = 2
    phrase = "This is a triggered phrase."

    logits_processors = [
        TriggerPhraseLogitsProcessor(
            llm_runner.tokenizer,
            batch_size=len(example_prompts),
            phrase=phrase,
            trigger_time=trigger_time,
            trigger_after=True,
        )
    ]

    processed_gen_output = llm_runner.generate_response(example_prompts, logits_processors, max_new_tokens=1000)
    assert phrase in processed_gen_output[0]
