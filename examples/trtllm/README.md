# Test TensorRT-LLM logits processors

## Quick Start

It's recommended to use [TensorRT-LLM release containers](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags) (>= 0.20.0) that has TensorRT-LLM pre-installed.
Alternatively, please follow [this documentation](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) to install it in [NGC PyTorch containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) (>=25.04).

## Examples

### GenLengthLogitsProcessor
A logits processor that adjusts the likelihood of the end-of-sequence (EOS) token based on the length of the generated sequence, encouraging or discouraging shorter answers.
```
python examples/trtllm/gen_length_logits_processor.py 
```

### CiteFromPromptLogitsProcessor
A logits processor which boosts or diminishes the likelihood of tokens present in the prompt (and optionally EOS token) to encourage the model to generate tokens similar to those seen in the prompt or vice versa.
```
python examples/trtllm/cite_prompt_logits_processor.py -p "Retrieved information:
    Pokémon is a Japanese media franchise consisting of video games, animated series and films, a trading card game, and other related media. 
    The franchise takes place in a shared universe in which humans co-exist with creatures known as Pokémon, a large variety of species endowed with special powers. 
    The franchise's target audience is children aged 5 to 12, but it is known to attract people of all ages.
    
    Can you shortly describe what Pokémon is?"
```

### ForceLastPhraseLogitsProcessor
A logits processor which forces LLMs to use the given phrase before they finalize their answers. Most common use cases can be providing references, thanking user with context etc.
```
python examples/trtllm/last_phrase_logits_processor.py
```

### MultipleChoiceLogitsProcessor
A logits processor to answer multiple choice questions with one of the choices.
```
python examples/trtllm/multiple_choice_logits_processor.py -p "I am getting a lot of calls during the day. What is more important for me to consider when I buy a new phone?
0. Camera
1. Screen resolution
2. Operating System
3. Battery"
```

### TriggerPhraseLogitsProcessor
A logits processor which triggers phrases when it encounters a given token.
```
python examples/trtllm/trigger_phrase_logits_processor.py -p "Generate a python function to calculate nth fibonacci number. Make it recursive. Keep thinking short."
```

### PreventHallucinationLogitsProcessor
A logits processor that mitigates hallucinated model outputs by enforcing a predefined fallback phrase when token confidence falls below a specified threshold.
```
python examples/trtllm/prevent_hallucination_logits_processor.py -p "What are Nobel Prizes? Name the winners in 1977"
```
