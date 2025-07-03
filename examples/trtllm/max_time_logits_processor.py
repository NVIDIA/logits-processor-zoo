from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import MaxTimeLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    lp = MaxTimeLogitsProcessor(tokenizer, boost_factor=1.0, complete_sentences=True, max_time=100)
    llm_tester.run([args.prompt], logits_processor=lp)

    lp = MaxTimeLogitsProcessor(tokenizer, boost_factor=-1.0, p=0, complete_sentences=True, max_time=1.0)
    llm_tester.run([args.prompt], logits_processor=lp)
