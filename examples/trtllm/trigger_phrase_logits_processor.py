from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import TriggerPhraseLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    lp = TriggerPhraseLogitsProcessor("...Wait, let me think more.", " function", tokenizer,
                                      trigger_count=2, trigger_after=False)
    llm_tester.run([args.prompt], logits_processor=lp)

    lp = TriggerPhraseLogitsProcessor("\n```python", " function", tokenizer, trigger_count=1, trigger_after=True)
    llm_tester.run([args.prompt], logits_processor=lp)
