import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList


class LLMRunner:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

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

        input_ids = self.tokenizer(prompts_with_template, return_tensors='pt', padding=True)["input_ids"]
        out_ids = self.model.generate(input_ids.cuda(), max_new_tokens=max_tokens, min_new_tokens=1, do_sample=False,
                                      logits_processor=LogitsProcessorList(logits_processor_list),
                                      temperature=None, top_p=None, top_k=None)
        gen_output = self.tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
        for prompt, out in zip(prompts, gen_output):
            print(f"Prompt: {prompt}")
            print()
            print(f"LLM response:\n{out.strip()}")
            print("-----END-----")
            print()
