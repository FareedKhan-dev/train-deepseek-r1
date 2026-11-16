# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-16

The first tagged release. On current libraries the notebook stopped at its first cell. It now runs from the top with the versions pinned in `requirements.txt`, and it was executed from start to finish on one RTX A6000, so every output in the notebook and the README comes from that run.

### Fixed

- **The notebook stopped at the imports.** TRL no longer exports `AutoModelForCausalLMWithValueHead`, `PPOConfig` or `PPOTrainer`, which the notebook imported but never used.
- **Both training configurations failed.** transformers removed `evaluation_strategy` (now `eval_strategy`), `overwrite_output_dir` and `warmup_ratio` (now a fraction passed to `warmup_steps`). GRPO is now configured with `GRPOConfig` directly instead of copying a `TrainingArguments`, and SFT with `SFTConfig`.
- **`SFTTrainer(tokenizer=...)`** is `processing_class=` in TRL. The README's version also passed `packing` and `max_seq_length` to `TrainingArguments`, which never accepted them.
- **The cosine reward crashed whenever it was enabled**: it required an `accuracy_rewards` argument that the trainer never passes. It now scores correctness itself with `accuracy_reward`, and completions longer than `max_len` no longer wrap around the cosine (a 2,000 character answer scored like an empty one).
- **The format reward accepted text after `</answer>`.** With `re.MULTILINE`, `$` matched at the end of any line, so a completion that kept writing after its answer still got the full reward. That is what the log in [#9](https://github.com/FareedKhan-dev/train-deepseek-r1/issues/9) shows: every completion ran to the 256 token limit while the format reward was 0.98.
- **The accuracy reward passed its arguments to `math_verify.verify()` the wrong way round.** The gold answer goes first; the comparison is not symmetric. Its `NormalizationConfig(equations=True)` is deprecated and printed a warning for every completion, 2,360 lines in one run.
- **The README's repetition penalty returned the factory instead of the reward function**, so the trainer would have called `get_repetition_penalty_reward(prompts=..., completions=...)`.
- **The logging callback printed the previous log entry**, or nothing on the first one ([#3](https://github.com/FareedKhan-dev/train-deepseek-r1/issues/3) was the crash this caused before), and its INFO messages never showed because logging was not configured. It now uses `on_log` and prints the mean of each reward too.
- **The inference helpers printed the whole conversation** (system prompt and question included) as the model's response. They now decode only the new tokens.
- **The README's example outputs were not produced by its code.** The few-shot prompt asked the target question twice, `refine_output` removed `umm...` from a text that says `ummm...`, and the dataset sample showed `from`/`value` keys that NuminaMath-TIR does not have. The few-shot, direct prompting and refinement examples are now notebook cells, and every output in the README is copied from the run.
- **GRPO stands for Group Relative Policy Optimization**, not "Gradient Reward Policy Optimization".

### Changed

- **GRPO learning rate 5e-5 to 1e-6** (TRL's default). With 5e-5 the accuracy on the 99 test problems fell from 16.6% to 0% within 100 steps and stayed below 1%.
- **The GRPO run is 300 steps** (`max_steps`); one epoch is 36,220 steps. It evaluates before training starts and every 50 steps, and the README reports what the run did: accuracy 16.6% before and 16.8% after, and a format reward near 0, because the base model almost never writes the tags.
- **vLLM generates the GRPO completions when it is installed** (Linux with an NVIDIA GPU), for the slowness reported in [#9](https://github.com/FareedKhan-dev/train-deepseek-r1/issues/9): 2.6 s per step instead of 13 s on an RTX A6000. Without it the notebook still runs, with `model.generate()`.
- `max_completion_length=512` and `num_generations=8` are set explicitly (TRL's old default cut completions at 256 tokens, see [#4](https://github.com/FareedKhan-dev/train-deepseek-r1/issues/4) and #9).
- The SFT stage runs 200 steps of 16 conversations, with `max_length=4096`.
- `requirements.txt` pins every library (PyTorch 2.14, transformers 5.20, TRL 1.18, vLLM 0.33 on Linux only) and drops `peft`, `wandb`, `torchvision` and `torchaudio`, which the notebook does not use, and the CUDA 11.8 index.
- The README explains the new settings, the Python and driver requirements, and why you should not export the notebook as `code.py` ([#6](https://github.com/FareedKhan-dev/train-deepseek-r1/issues/6)).

### Added

- `tests/test_notebook.py`: the notebook's reward functions called the way GRPOTrainer calls them, two GRPO steps and two SFT steps on CPU with a tiny model, checks that every configuration and trainer argument exists in the pinned libraries, that the notebook ran from the top without errors, and that the README shows the code the notebook runs. 15 of the 21 tests fail on the previous notebook.
- CI that runs the tests on every push and pull request.
- A cell that frees the GPU memory of the GRPO stage, and a test of the SFT model on the same problem as the GRPO model.

### Removed

- The `ModelConfig` dataclass, which nothing used (its `flash_attention_2` setting never reached the model).

[1.0.0]: https://github.com/FareedKhan-dev/train-deepseek-r1/releases/tag/v1.0.0
