# Security Policy: logits-processor-zoo

## Reporting a Vulnerability

If you discover a potential security vulnerability in logits-processor-zoo, please **do not open a public issue or pull request.** Public reports can expose exploit details before maintainers and NVIDIA PSIRT have had time to investigate and coordinate a fix.

Report potential vulnerabilities through one of these private channels:

- **NVIDIA Vulnerability Disclosure Program** (preferred): https://www.nvidia.com/en-us/security/
- **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - NVIDIA encourages use of the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key) for sensitive reports.
- **GitHub Private Vulnerability Reporting:** use this repository's **Security** tab and select **Report a vulnerability**.

Please include as much of the following information as possible:

- Product/project name, affected version, branch, or commit
- Vulnerability type and affected component, such as a logits processor implementation, framework adapter, example script, notebook, packaging, or CI workflow
- Step-by-step reproduction instructions
- Proof-of-concept code or configuration, if available
- Impact assessment, including whether the issue affects model output integrity, availability of generation, local code execution through model loading, or exposure of prompts/model outputs
- Environment details, including Python version, framework version, model/runtime backend, and whether the issue occurs with transformers, vLLM, or TensorRT-LLM

Detailed reports help NVIDIA evaluate and address issues faster. NVIDIA's PSIRT team will acknowledge receipt, validate the vulnerability and severity, coordinate fixes with maintainers, and publish security bulletins or advisories as appropriate.

## Security Architecture & Context

logits-processor-zoo is a Python package that provides custom logits processors for LLM generation workflows. The repository includes framework-specific implementations for Hugging Face transformers, vLLM, and TensorRT-LLM, plus example scripts and notebooks under `lpz_examples/`.

This software operates primarily as a **Library / SDK** for inference-time generation control. Its primary security responsibility is to preserve the integrity and expected behavior of token selection when callers attach processors such as `CiteFromPromptLogitsProcessor`, `ForceLastPhraseLogitsProcessor`, `MultipleChoiceLogitsProcessor`, `TriggerPhraseLogitsProcessor`, `PreventHallucinationLogitsProcessor`, `GenLengthLogitsProcessor`, and `MaxTimeLogitsProcessor` to model generation.

**Repository Exposure Classification:** Not determined.
Basis: origin remote is on github.com, but source visibility was not confirmed; this document is written to public-safe detail.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: user-confirmed override; the project is an externally distributed Python package for LLM inference integrations across transformers, vLLM, and TensorRT-LLM.

The main trust boundaries are:

- **Caller-to-library boundary:** Application code supplies prompts, tokenizer objects or tokenizer names, logits processor parameters, and generation settings. The package assumes callers understand the trust level of those inputs.
- **Tokenizer/model boundary:** Several implementations and examples rely on tokenizer or model loading through framework APIs, including `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, and backend-specific `LLM(...)` constructors. Model and tokenizer sources are outside this package's control.
- **Generation-integrity boundary:** Processor classes directly mutate logits or scores tensors during decoding. Incorrect parameters, invalid tokenization assumptions, or malformed tensor shapes can change model output behavior or fail generation.
- **Example-code boundary:** Scripts and notebooks in `lpz_examples/` accept model names and prompts for local experimentation. They are not authentication or authorization boundaries and should not be treated as production service front ends.
- **CI/package boundary:** The GitHub Actions workflow installs the package and runs lint/tests with read-only repository contents permission. Package consumers remain responsible for dependency pinning, trusted package sources, and runtime isolation.

The repository does not define an HTTP API, gRPC service, database connection, secrets store, session management layer, authentication handler, authorization policy, TLS configuration, or network listener. If logits-processor-zoo is used inside a network-facing product or service, those security controls must be provided by the embedding application and deployment environment.

### Threat Model

The following scenarios represent the primary security concerns for this project, based on repository structure, dependencies, examples, and code paths:

1. **Untrusted Model or Tokenizer Loading:** The README and examples demonstrate loading models and tokenizers through framework APIs, including `trust_remote_code=True` in vLLM usage examples and model-name arguments in `lpz_examples/trtllm/utils.py`. If a user loads an untrusted model or tokenizer source, the framework may execute code or load artifacts outside logits-processor-zoo's control.

2. **Generation Integrity Bypass Through Logits Mutation:** The processor implementations in `logits_processor_zoo/transformers/`, `logits_processor_zoo/vllm/`, and `logits_processor_zoo/trtllm/` directly modify score/logit tensors. Misconfigured processor parameters, unexpected tokenization behavior, or incompatible backend tensor shapes could force unintended tokens, suppress expected output, or weaken application-level output constraints.

3. **Prompt-Influenced Citation or Token Boosting:** `CiteFromPromptLogitsProcessor` boosts tokens found in the prompt and can apply conditional boosts based on prior generated tokens. In applications where prompts include untrusted user content, a caller that treats this processor as a security control could unintentionally increase the likelihood of reproducing attacker-provided prompt text.

4. **Availability Impact From Forced Phrase or Time-Based Processors:** `ForceLastPhraseLogitsProcessor`, `TriggerPhraseLogitsProcessor`, `PreventHallucinationLogitsProcessor`, and `MaxTimeLogitsProcessor` can force token sequences or alter EOS behavior. Long phrases, aggressive thresholds, repeated triggers, or backend-specific state handling can increase latency, truncate useful output, or cause generation failures in downstream applications.

5. **Example Script Misuse In Production Contexts:** Example utilities under `lpz_examples/` accept user-controlled prompts and model names through command-line arguments and load model artifacts for local demonstration. If copied into a service without additional validation, sandboxing, and resource controls, these examples could expose model-loading and prompt-handling risks to untrusted callers.

6. **Dependency and Runtime Supply Chain Exposure:** The package depends on PyTorch and transformers, with optional vLLM integration and TensorRT-LLM examples. Vulnerabilities or unsafe defaults in those runtimes, model repositories, tokenizers, or package distribution channels can affect consumers even when logits-processor-zoo code is unchanged.

### Critical Security Assumptions

- **Trusted model and tokenizer sources:** The package assumes callers load models, tokenizers, and framework runtime code only from trusted sources, especially when using examples or configurations that enable remote model code.
- **Embedding application provides authentication and authorization:** logits-processor-zoo does not authenticate users, authorize prompts, enforce tenant boundaries, or manage sessions. Any service exposing generation to users must implement those controls outside this library.
- **Embedding application validates prompts and processor parameters:** The library assumes callers validate prompt content, phrase strings, thresholds, token choices, batch sizes, model names, and generation limits before passing them into processors or examples.
- **Runtime frameworks enforce tensor and memory safety:** The processors assume PyTorch, transformers, vLLM, TensorRT-LLM, CUDA, and the host OS correctly enforce process isolation, memory safety guarantees, and device access controls.
- **Resource limits are enforced by the caller or deployment environment:** The library does not provide rate limiting, quotas, request cancellation, GPU isolation, or denial-of-service protection for long prompts, large batches, long forced phrases, or expensive model loads.
- **Transport security is outside this package:** The repository contains no network listener or TLS configuration. Any API, notebook server, or inference service embedding this package must provide TLS and network access control at the service or infrastructure layer.
- **Examples are non-production starting points:** The scripts and notebooks under `lpz_examples/` are intended for demonstration and validation. Production use requires separate hardening, dependency pinning, logging policy, secrets handling, and operational monitoring.

## Dependency Security

logits-processor-zoo is packaged with Poetry and declares Python `>=3.10`, `torch`, `transformers >=4.41.2`, `accelerate >=0.26.1`, and optional `vllm >=0.5.0.post1`. Consumers should:

- Pin and regularly update runtime dependencies in deployable environments.
- Monitor advisories for PyTorch, transformers, vLLM, TensorRT-LLM, CUDA/container images, and tokenizer/model-loading libraries.
- Load model artifacts from trusted registries or vetted local paths.
- Avoid enabling remote model code unless the model source is trusted and the runtime is appropriately isolated.

## Deployment Guidance

When embedding logits-processor-zoo in an application or service:

- Treat all prompts, model names, tokenizer names, and processor configuration from users as untrusted input.
- Validate processor parameters against application policy before generation.
- Apply model and tokenizer allowlists for production deployments.
- Run inference workloads with least privilege and appropriate GPU/container isolation.
- Enforce authentication, authorization, rate limits, request timeouts, and audit logging at the embedding service layer.
- Keep example notebooks and scripts out of production paths unless they have been reviewed and hardened.
