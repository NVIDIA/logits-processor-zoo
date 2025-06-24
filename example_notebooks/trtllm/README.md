# Test TensorRT-LLM logits processors

## Quick Start

Follow this guide to create an engine:
https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html

## Examples

```
python example_notebooks/trtllm/gen_length_logits_processor.py --backend pytorch --model_name Qwen/Qwen2.5-1.5B-Instruct
python example_notebooks/trtllm/multiple_choice_logits_processor.py --model_name Qwen/Qwen2.5-1.5B-Instruct --prompt "Which one is heavier?\n1. 1 kg\n2. 100 kg\n3. 10 kg\nAnswer:"
```