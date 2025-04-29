from logits_processor_zoo.utils import text_to_token, get_new_line_tokens, enforce_tokens
import torch


def test_text_to_token(llm_runner):
    assert text_to_token(llm_runner.tokenizer, ",", last=False) == 1919
    assert text_to_token(llm_runner.tokenizer, "apple, orange,", last=True) == 29892
    assert text_to_token(llm_runner.tokenizer, "apple, orange\n", last=True) == 13

    try:
        token = text_to_token(llm_runner.tokenizer, "apple, orange,", last=False)
    except Exception:
        token = -1

    assert token == -1


def test_get_new_line_tokens(llm_runner):
    assert get_new_line_tokens(llm_runner.tokenizer) == {13}


def test_enforce_tokens():
    scores = torch.FloatTensor([0.1, -0.4, -0.2, -0.6, 1.1])
    tokens = [1, 2]

    scores = enforce_tokens(scores, tokens)
    _, top2_tokens = torch.topk(scores, k=2)
    assert torch.equal(top2_tokens, torch.tensor([2, 1]))
