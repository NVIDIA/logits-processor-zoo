import argparse
from typing import List
from tensorrt_llm.sampling_params import SamplingParams, LogitsProcessor


class TRTLLMTester:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        # Temporarily attempt to import the torch backend until it becomes default
        try:
            from tensorrt_llm._torch import LLM
        except ImportError:
            from tensorrt_llm import LLM

        self.llm = LLM(model=model_name)

    def run(self, prompts: List[str], max_tokens: int = 256, logits_processor: LogitsProcessor = None):
        sparams = {"top_k": 1, "max_tokens": max_tokens, "temperature": 0.001}
        if logits_processor:
            sparams["logits_processor"] = logits_processor

        prompts_with_template = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            text = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts_with_template.append(text)

        gens = self.llm.generate(prompts_with_template, SamplingParams(**sparams))
        for prompt, gen in zip(prompts, gens):
            print(prompt)
            print(gen.outputs[0].text)


def get_parser():
    parser = argparse.ArgumentParser(description="Logits Processor Example")
    parser.add_argument("--model_name",
                        "-m",
                        type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Directory or HF link containing model")
    parser.add_argument("--prompt",
                        "-p",
                        type=str,
                        default="Please give me information about macaques:",
                        help="Prompt to test")

    return parser.parse_args()
