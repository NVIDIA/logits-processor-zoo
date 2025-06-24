import argparse
from typing import List
from tensorrt_llm.sampling_params import SamplingParams, LogitsProcessor

class TRTLLMTester:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", backend: str = "tensorrt-llm",
                 logits_processor: LogitsProcessor = None):
        if backend == "pytorch":
            from tensorrt_llm._torch import LLM
        else:
            from tensorrt_llm import LLM

        self.llm = LLM(model=model_name)
        self.lp = logits_processor

    def run(self, prompts: List[str], max_tokens: int = 256):
        sparams = {"top_k": 1, "max_tokens": max_tokens, "temperature": 0.001}
        if self.lp:
            sparams["logits_processor"] = self.lp
        output = self.llm.generate(prompts, SamplingParams(**sparams))
        print(output)


def get_parser():
    parser = argparse.ArgumentParser(description="Logits Processor Example")
    parser.add_argument("--model_name",
                        "-m",
                        type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Directory or HF link containing model")
    parser.add_argument("--backend",
                        "-b",
                        type=str,
                        default="tensorrt-llm",
                        help="TensorRT-LLM backend")
    parser.add_argument("--prompt",
                        "-p",
                        type=str,
                        default="Please give me information about macaques:",
                        help="Prompt to test")

    return parser.parse_args()
