import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList


class LLMRunner:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate_response(self, prompts, logits_processor_list=None, max_tokens=1000):
        if logits_processor_list is None:
            logits_processor_list = []

        for prompt in prompts:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            inputs = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=1,
                logits_processor=LogitsProcessorList(logits_processor_list),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

            gen_output = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # Extract only the generated output after the original input length
            generated_text = gen_output[0][
                len(
                    self.tokenizer.decode(
                        inputs["input_ids"][0], skip_special_tokens=True
                    )
                ):
            ].strip()

            print(f"Prompt: {prompt}")
            print()
            print(f"LLM response:\n{generated_text}")
            print("-----END-----")
            print()
            print()
