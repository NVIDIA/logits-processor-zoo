from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import TriggerPhraseLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    lp = TriggerPhraseLogitsProcessor(
        tokenizer, "...Wait, let me think more.", " function", trigger_count=2, trigger_after=False
    )
    llm_tester.run([args.prompt], logits_processor=lp)

    lp = TriggerPhraseLogitsProcessor(tokenizer, "\n```python", " function", trigger_count=1, trigger_after=True)
    llm_tester.run([args.prompt], logits_processor=lp)

    lp = TriggerPhraseLogitsProcessor(
        tokenizer, "<interruption> only a few seconds left...", trigger_time=2, trigger_count=1, trigger_after=True
    )
    llm_tester.run([args.prompt], logits_processor=lp)
