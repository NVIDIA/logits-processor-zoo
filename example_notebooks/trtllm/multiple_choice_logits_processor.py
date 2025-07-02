from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import MultipleChoiceLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    lp = MultipleChoiceLogitsProcessor(tokenizer, choices=["0", "1", "2", "3"])
    llm_tester.run([args.prompt], logits_processor=lp, max_tokens=1)

    lp = MultipleChoiceLogitsProcessor(tokenizer, choices=["0", "1", "2", "3"], delimiter=".", boost_first_words=2.0)
    llm_tester.run([args.prompt], logits_processor=lp, max_tokens=1)
