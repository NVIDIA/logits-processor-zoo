# Test TensorRT-LLM logits processors

## Quick Start

Follow this guide to create an engine:
https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html

## Examples

```
python example_notebooks/trtllm/gen_length_logits_processor.py 

python example_notebooks/trtllm/cite_prompt_logits_processor.py -p "Retrieved information:
    Pokémon is a Japanese media franchise consisting of video games, animated series and films, a trading card game, and other related media. 
    The franchise takes place in a shared universe in which humans co-exist with creatures known as Pokémon, a large variety of species endowed with special powers. 
    The franchise's target audience is children aged 5 to 12, but it is known to attract people of all ages.
    
    Can you shortly describe what Pokémon is?"

python example_notebooks/trtllm/last_phrase_logits_processor.py

python example_notebooks/trtllm/multiple_choice_logits_processor.py -p "I am getting a lot of calls during the day. What is more important for me to consider when I buy a new phone?
0. Camera
1. Screen resolution
2. Operating System
3. Battery"

python example_notebooks/trtllm/prevent_hallucination_logits_processor.py -p "Tell me the Nobel Prizes in 1977"
```