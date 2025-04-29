import vllm


class vLLMRunner:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model = vllm.LLM(
            model_name,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True
        )
        self.tokenizer = self.model.get_tokenizer()

    def generate_response(self, prompts, logits_processor_list=None, max_tokens=1000):
        if logits_processor_list is None:
            logits_processor_list = []

        prompts_with_template = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts_with_template.append(text)

        gen_output = self.model.generate(
            prompts_with_template,
            vllm.SamplingParams(
                n=1,
                temperature=0,
                seed=0,
                skip_special_tokens=True,
                max_tokens=max_tokens,
                logits_processors=logits_processor_list
            ),
            use_tqdm=False
        )

        for prompt, out in zip(prompts, gen_output):
            out = out.outputs[0].text
            print(f"Prompt: {prompt}")
            print(out)
            print("-----END-----")
            print()
