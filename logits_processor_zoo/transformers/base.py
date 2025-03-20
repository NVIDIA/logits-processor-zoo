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

import torch


class BaseLogitsProcessor:
    def __init__(self):
        self.prompt_token_ids = None
        self.prev_token_ids = None

    def _reset(self):
        pass

    def _check_new_generation(self, input_ids: torch.LongTensor):
        first_time = self.prompt_token_ids is None
        if first_time:
            self._reset()
            self.prompt_token_ids = input_ids
        else:
            same_gen = False
            if input_ids.shape[1] > 1:
                same_gen = torch.equal(input_ids[:, :-1], self.prev_token_ids)

            if not same_gen:
                self._reset()
                self.prompt_token_ids = input_ids

        self.prev_token_ids = input_ids

    def _process(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        self._check_new_generation(input_ids)
        scores = self._process(input_ids, scores)
        return scores
