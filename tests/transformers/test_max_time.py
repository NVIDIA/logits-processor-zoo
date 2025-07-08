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

import time
from logits_processor_zoo.transformers import MaxTimeLogitsProcessor


def test_max_time_logits_processor(llm_runner):
    """Test that the phrase is triggered when the specified token is generated."""
    example_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]

    max_time = 2
    tolerance = 1
    start_time = time.time()

    logits_processors = [MaxTimeLogitsProcessor(llm_runner.tokenizer, max_time=max_time, complete_sentences=False)]
    outs = llm_runner.generate_response(example_prompts, logits_processors, max_new_tokens=1000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(outs)
    assert elapsed_time <= max_time + tolerance
