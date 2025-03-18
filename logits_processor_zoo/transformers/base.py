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
        self.input_len = None

    def _reset(self):
        pass

    def _reset_if_new_batch(self, input_ids: torch.LongTensor):
        if self.input_len is not None:
            if input_ids.shape[1] != self.input_len + 1:
                self._reset()
        else:
            self._reset()

        self.input_len = input_ids.shape[1]

    def _process(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        self._reset_if_new_batch(input_ids)
        scores = self._process(input_ids, scores)
        return scores
