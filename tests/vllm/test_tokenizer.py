import pytest

from logits_processor_zoo.vllm import tokenizer as tokenizer_module
from logits_processor_zoo.vllm.tokenizer import TOKENIZER_ALLOWLIST_ENV_VAR, get_vllm_tokenizer


def test_tokenizer_object_is_returned_without_allowlist(monkeypatch):
    tokenizer = object()
    monkeypatch.delenv(TOKENIZER_ALLOWLIST_ENV_VAR, raising=False)

    assert get_vllm_tokenizer(tokenizer) is tokenizer


def test_string_tokenizer_is_disabled_without_allowlist(monkeypatch):
    def fail_from_pretrained(*args, **kwargs):
        raise AssertionError("from_pretrained should not be called")

    monkeypatch.delenv(TOKENIZER_ALLOWLIST_ENV_VAR, raising=False)
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", fail_from_pretrained)

    with pytest.raises(ValueError, match="disabled"):
        get_vllm_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")


def test_unallowlisted_string_tokenizer_is_rejected_before_loading(monkeypatch):
    def fail_from_pretrained(*args, **kwargs):
        raise AssertionError("from_pretrained should not be called")

    monkeypatch.setenv(TOKENIZER_ALLOWLIST_ENV_VAR, "trusted/model")
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", fail_from_pretrained)

    with pytest.raises(ValueError, match="not allowlisted"):
        get_vllm_tokenizer("attacker/model")


def test_allowlisted_string_tokenizer_loads_from_local_files_only(monkeypatch):
    tokenizer = object()
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return tokenizer

    monkeypatch.setenv(TOKENIZER_ALLOWLIST_ENV_VAR, "Qwen/Qwen2.5-1.5B-Instruct,other/model")
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", from_pretrained)

    assert get_vllm_tokenizer("Qwen/Qwen2.5-1.5B-Instruct") is tokenizer
    assert calls == [("Qwen/Qwen2.5-1.5B-Instruct", {"local_files_only": True})]


def test_allowlisted_tokenizer_urls_are_rejected_before_loading(monkeypatch):
    def fail_from_pretrained(*args, **kwargs):
        raise AssertionError("from_pretrained should not be called")

    monkeypatch.setenv(TOKENIZER_ALLOWLIST_ENV_VAR, "https://huggingface.co/attacker/model")
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", fail_from_pretrained)

    with pytest.raises(ValueError, match="URLs are not allowed"):
        get_vllm_tokenizer("https://huggingface.co/attacker/model")
