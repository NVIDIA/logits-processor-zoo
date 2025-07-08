from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import CiteFromPromptLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_tester = TRTLLMTester(args.model_name)

    lp = CiteFromPromptLogitsProcessor(tokenizer, boost_factor=1.0, boost_eos=False, conditional_boost_factor=3.0)
    llm_tester.run([args.prompt], logits_processor=lp)

    lp = CiteFromPromptLogitsProcessor(tokenizer, boost_factor=-1.0, boost_eos=False, conditional_boost_factor=-1.0)
    llm_tester.run([args.prompt], logits_processor=lp)
