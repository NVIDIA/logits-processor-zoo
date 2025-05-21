from typing import List
import torch
from transformers import PreTrainedTokenizer
from logits_processor_zoo.utils import text_to_token, enforce_tokens


class TextRetrievalLogitsProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, doc: str, split_by_line: bool = True):
        self.tokenizer = tokenizer
        self.split_by_line = split_by_line
        self.doc = doc
        if self.split_by_line:
            self.sep_token = text_to_token(tokenizer, "It is a new line\n", last=True)
        else:
            self.sep_token = text_to_token(tokenizer, "It is a sentence.", last=True)

        self.doc_tokens = self.tokenizer.encode(self.doc, add_special_tokens=False)
        self._init_start_tokens()

    def _init_start_tokens(self):
        self.start_tokens = [self.doc_tokens[0]]

        prev_token = self.doc_tokens[0]
        for token in self.doc_tokens[1:]:
            if prev_token == self.sep_token:
                self.start_tokens.append(token)
            prev_token = token

    def clone(self):
        return TextRetrievalLogitsProcessor(self.tokenizer, self.doc, self.split_by_line)

    def _find_all_next_tokens(self, past_token_ids):
        gen_len = len(past_token_ids)
        return [self.doc_tokens[i + gen_len] for i in range(len(self.doc_tokens) - gen_len + 1)
                if (self.doc_tokens[i:i + gen_len] == past_token_ids) and (i + gen_len < len(self.doc_tokens))]

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        if not past_token_ids:
            next_tokens = self.start_tokens
        else:
            next_tokens = self._find_all_next_tokens(list(past_token_ids))

            if (past_token_ids[-1] == self.sep_token) or (len(next_tokens) == 0):
                next_tokens.append(self.tokenizer.eos_token_id)

        scores = enforce_tokens(scores, next_tokens)
        return scores
