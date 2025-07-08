from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import PreventHallucinationLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    lp = PreventHallucinationLogitsProcessor(tokenizer, minp=0.25, tolerate=1)
    llm_tester.run([args.prompt], logits_processor=lp)
