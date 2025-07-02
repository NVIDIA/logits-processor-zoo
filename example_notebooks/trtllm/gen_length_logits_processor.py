from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import GenLengthLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    lp = GenLengthLogitsProcessor(tokenizer, boost_factor=1.0, complete_sentences=True)
    llm_tester.run([args.prompt], logits_processor=lp)

    lp = GenLengthLogitsProcessor(tokenizer, boost_factor=-1.0, p=0, complete_sentences=True)
    llm_tester.run([args.prompt], logits_processor=lp)
