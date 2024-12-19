from logits_processor_zoo.utils import text_to_token, get_new_line_tokens


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
