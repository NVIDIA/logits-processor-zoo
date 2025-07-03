from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import ForceLastPhraseLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    phrase = "\n\nThanks for trying our application! If you have more questions about"
    lp = ForceLastPhraseLogitsProcessor(phrase, tokenizer)

    llm_tester.run([args.prompt], logits_processor=lp)
