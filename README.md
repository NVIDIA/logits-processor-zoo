[![PyPI version](https://badge.fury.io/py/logits-processor-zoo.svg)](https://badge.fury.io/py/logits-processor-zoo)
[![License: MIT](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/licenses/Apache2.0)

<p align="center">
    <img src="docs/logo.jpg" width="50%">
</p>

# logits-processor-zoo

Struggling to get LLMs to follow your instructions? LogitsProcessorZoo offers a zoo of tools to use LLMs for specific tasks, beyond just grammar enforcement!

## Installation

```bash
pip install logits-processor-zoo
```

## Supported Frameworks
* transformers
* vLLM
* TensorRT-LLM (>=0.20.0)

## Usage

```python
import vllm
from logits_processor_zoo.vllm import GenLengthLogitsProcessor, CiteFromPromptLogitsProcessor, ForceLastPhraseLogitsProcessor

model = vllm.LLM(
            model_name,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True
        )
tokenizer = model.get_tokenizer()
        
logits_processors = [
    CiteFromPromptLogitsProcessor(tokenizer, boost_factor=2.0),
    GenLengthLogitsProcessor(tokenizer, boost_factor=-0.2, p=1),
    ForceLastPhraseLogitsProcessor("\n\nReferences:\n", tokenizer)
]


gen_output = model.generate(
            prompts,
            vllm.SamplingParams(
                n=1,
                temperature=0,
                seed=0,
                skip_special_tokens=True,
                max_tokens=64,
                logits_processors=logits_processors
            ),
            use_tqdm=False
        )
```


For the detailed examples in each framework, please have a look at **example_notebook** directory.

## Available Logits Processors

### GenLengthLogitsProcessor
A logits processor that adjusts the likelihood of the end-of-sequence (EOS) token based on the length of the generated sequence, encouraging or discouraging shorter answers.

### CiteFromPromptLogitsProcessor
A logits processor which boosts or diminishes the likelihood of tokens present in the prompt (and optionally EOS token) to encourage the model to generate tokens similar to those seen in the prompt or vice versa.

### ForceLastPhraseLogitsProcessor
A logits processor which forces LLMs to use the given phrase before they finalize their answers. Most common use cases can be providing references, thanking user with context etc.

### MultipleChoiceLogitsProcessor
A logits processor to answer multiple choice questions with one of the choices. A multiple choice question is like:
```
I am getting a lot of calls during the day. What is more important for me to consider when I buy a new phone?
0. Camera
1. Screen resolution
2. Operating System
3. Battery
```
The goal is to make LLM generate "3" as an answer.

### TriggerPhraseLogitsProcessor
A logits processor which triggers phrases when it encounters a given token or after a specified time.
One common use case is to force writing python code just after thinking:
```python
trigger_python = TriggerPhraseLogitsProcessor(phrase="\n```python", trigger_token_phrase="</think>", 
                                              tokenizer=tokenizer, trigger_count=1, trigger_after=True)
```
### PreventHallucinationLogitsProcessor
A logits processor that mitigates hallucinated model outputs by enforcing a predefined fallback phrase when token confidence falls below a specified threshold.

### MaxTimeLogitsProcessor
A logits processor that enforces the end-of-sentence (EOS) token after a specified maximum time passes, optionally waiting for a new line or a full stop. Useful for controlling generation time and ensuring responses complete within time constraints.