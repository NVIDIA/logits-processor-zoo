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

import os
from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizer


TOKENIZER_ALLOWLIST_ENV_VAR = "LOGITS_PROCESSOR_ZOO_VLLM_ALLOWED_TOKENIZERS"


def get_vllm_tokenizer(tokenizer: Union[PreTrainedTokenizer, str]) -> PreTrainedTokenizer:
    if not isinstance(tokenizer, str):
        return tokenizer

    _validate_tokenizer_name(tokenizer)
    return AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)


def _validate_tokenizer_name(tokenizer_name: str):
    allowed_tokenizers = {
        allowed.strip()
        for allowed in os.environ.get(TOKENIZER_ALLOWLIST_ENV_VAR, "").split(",")
        if allowed.strip()
    }

    if not allowed_tokenizers:
        raise ValueError(
            "String tokenizer loading is disabled for vLLM logits processors. "
            f"Pass a tokenizer object, or set {TOKENIZER_ALLOWLIST_ENV_VAR} to trusted tokenizer names."
        )

    if not tokenizer_name or tokenizer_name != tokenizer_name.strip():
        raise ValueError("Tokenizer names must be non-empty and must not include surrounding whitespace.")

    if "://" in tokenizer_name:
        raise ValueError("Tokenizer URLs are not allowed.")

    if tokenizer_name not in allowed_tokenizers:
        raise ValueError(
            "Tokenizer is not allowlisted for vLLM logits processors. "
            f"Add trusted tokenizer names to {TOKENIZER_ALLOWLIST_ENV_VAR} on the server."
        )
