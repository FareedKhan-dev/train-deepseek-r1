"""Tests for code.ipynb.

The code under test is read from the notebook itself, so these tests check exactly what readers run.
Set NOTEBOOK=path/to/other.ipynb to run them against another version of the notebook.
"""

import ast
import inspect
import json
import logging
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(os.environ.get("NOTEBOOK", ROOT / "code.ipynb"))
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

# What the reward cells need from the notebook's import cell
PREAMBLE = """
import math, re
from dataclasses import dataclass, field
from typing import List, Optional
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
"""


def notebook_cells(kind="code"):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == kind]


def code_sources():
    return ["".join(c["source"]) for c in notebook_cells()]


def cell_defining(name):
    found = [src for src in code_sources() if re.search(rf"^(def|class) {name}\b", src, re.M)]
    assert len(found) == 1, f"{name} is defined in {len(found)} cells"
    return found[0]


def calls_in_notebook(func_name):
    """Every call to func_name in the notebook, as (cell source, ast.Call)."""
    found = []
    for src in code_sources():
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == func_name:
                found.append((src, node))
    return found


@pytest.fixture(scope="module")
def nb():
    """The notebook's reward functions and reward settings, in a fresh namespace."""
    ns = {"__name__": "__main__"}
    exec(PREAMBLE, ns)
    for name in ("accuracy_reward", "format_reward", "reasoning_steps_reward", "get_cosine_scaled_reward",
                 "get_repetition_penalty_reward", "GRPOScriptArguments", "get_reward_functions"):
        exec(cell_defining(name), ns)
    return ns


def call_like_trl(func, contents, solutions):
    """Call a reward function the way GRPOTrainer does (TRL 1.18, GRPOTrainer._calculate_rewards)."""
    n = len(contents)
    return func(
        prompts=[[{"role": "user", "content": "What is 2 + 3 * 4?"}]] * n,
        completions=[[{"role": "assistant", "content": c}] for c in contents],
        completion_ids=[[0]] * n,
        problem=["What is 2 + 3 * 4?"] * n,
        solution=solutions,
        trainer_state=None,
        log_extra=lambda *args: None,
        log_metric=lambda *args: None,
    )


GOLD = r"Multiply first, then add. The answer is $\boxed{14}$."
RIGHT = "<think>3 * 4 = 12, then 2 + 12 = 14</think>\n<answer>\\boxed{14}</answer>"
WRONG = "<think>2 + 3 = 5, then 5 * 4 = 20</think>\n<answer>\\boxed{20}</answer>"


# ----------------------------------------------------------------------------- reward functions

def test_accuracy_reward(nb):
    rewards = call_like_trl(nb["accuracy_reward"], [RIGHT, WRONG, RIGHT], [GOLD, GOLD, "no math in here"])
    assert rewards == [1.0, 0.0, 0.5]


def test_accuracy_reward_passes_the_gold_answer_first(nb, monkeypatch):
    # math_verify.verify(gold, target) is not symmetric, the gold answer must come first
    seen = []
    monkeypatch.setitem(nb, "verify", lambda first, second, **kwargs: seen.append((first, second)) or True)
    call_like_trl(nb["accuracy_reward"], [WRONG], [GOLD])
    gold = nb["parse"](GOLD, extraction_mode="first_match", extraction_config=[nb["LatexExtractionConfig"]()])
    assert [str(x) for x in seen[0][0]] == [str(x) for x in gold]


def test_format_reward(nb):
    contents = [
        "<think>3 * 4 = 12</think>\n<answer>14</answer>",
        "<think>3 * 4 = 12</think><answer>14</answer>\n",  # a final newline is fine
        "<answer>14</answer>",  # no reasoning
        "<think>3 * 4 = 12</think>\n<answer>14</answer>\nAnd here is some more text.",  # keeps going
        "Sure! <think>3 * 4 = 12</think>\n<answer>14</answer>",  # does not start with <think>
    ]
    assert call_like_trl(nb["format_reward"], contents, [GOLD] * 5) == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_reasoning_steps_reward(nb):
    contents = ["Step 1: multiply.\nStep 2: add.\nStep 3: check.", "First, multiply.", "14"]
    assert call_like_trl(nb["reasoning_steps_reward"], contents, [GOLD] * 3) == [1.0, pytest.approx(1 / 3), 0.0]


def test_cosine_reward_is_called_like_any_other_reward(nb):
    cosine = nb["get_cosine_scaled_reward"](max_len=1000)
    rewards = call_like_trl(cosine, [RIGHT, WRONG, RIGHT], [GOLD, GOLD, "no math in here"])
    assert len(rewards) == 3
    assert rewards[0] > 0.8 and rewards[1] < -0.1 and rewards[2] == 0.0


def test_cosine_reward_prefers_short_right_answers_and_long_wrong_ones(nb):
    cosine = nb["get_cosine_scaled_reward"](max_len=1000)
    padding = " Let me double check that." * 20  # about 500 characters
    short_right, long_right = call_like_trl(cosine, [RIGHT, RIGHT.replace("</think>", padding + "</think>")],
                                            [GOLD, GOLD])
    short_wrong, long_wrong = call_like_trl(cosine, [WRONG, WRONG.replace("</think>", padding + "</think>")],
                                            [GOLD, GOLD])
    assert short_right > long_right > 0.8
    assert short_wrong < long_wrong < -0.1


def test_cosine_reward_does_not_wrap_around_past_max_len(nb):
    cosine = nb["get_cosine_scaled_reward"](max_len=100)
    at_max = RIGHT.replace("</think>", "x" * (100 - len(RIGHT)) + "</think>")
    twice_max = RIGHT.replace("</think>", "x" * (200 - len(RIGHT)) + "</think>")
    assert len(at_max) == 100 and len(twice_max) == 200
    r_at_max, r_twice_max = call_like_trl(cosine, [at_max, twice_max], [GOLD, GOLD])
    assert r_at_max == pytest.approx(0.8) and r_twice_max == pytest.approx(0.8)


def test_repetition_penalty_reward(nb):
    penalty = nb["get_repetition_penalty_reward"](ngram_size=3, max_penalty=-0.1)
    looping = "I think the answer is 14. " * 10
    varied = "First multiply three by four to get twelve, then add two and the total is fourteen."
    rewards = call_like_trl(penalty, [looping, varied, "14", ""], [GOLD] * 4)
    assert rewards[0] < -0.05 and rewards[1:] == [0.0, 0.0, 0.0]


def test_every_reward_in_the_registry_works_with_the_trainer(nb):
    names = ["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty"]
    funcs = nb["get_reward_functions"](nb["GRPOScriptArguments"](reward_funcs=names))
    assert len(funcs) == 5
    for func in funcs:
        rewards = call_like_trl(func, [RIGHT, WRONG], [GOLD, GOLD])
        assert len(rewards) == 2 and all(isinstance(r, float) for r in rewards), func


# ----------------------------------------------------------------------------- training setup

def test_the_import_cell_works_with_the_pinned_libraries():
    imports = [src for src in code_sources() if src.startswith("# Import necessary libraries")]
    assert len(imports) == 1
    exec(imports[0], {})


def keyword_values(call, ns):
    return {kw.arg: eval(compile(ast.Expression(kw.value), "<notebook>", "eval"), ns) for kw in call.keywords}


@pytest.mark.parametrize("config_name", ["TrainingArguments", "GRPOConfig", "SFTConfig"])
def test_every_training_config_in_the_notebook_is_accepted(config_name, tmp_path):
    import transformers
    import trl

    config_class = getattr(transformers, config_name, None) or getattr(trl, config_name)
    ns = {"OUTPUT_DIR": str(tmp_path), "USE_VLLM": False}
    for _, call in calls_in_notebook(config_name):
        kwargs = keyword_values(call, ns)
        kwargs.update(bf16=False, use_cpu=True)  # the CI machine has no GPU
        config_class(**kwargs)


def test_the_notebook_trains_with_grpo_and_sft():
    assert len(calls_in_notebook("GRPOConfig")) == 1 and len(calls_in_notebook("SFTConfig")) == 1


@pytest.mark.parametrize("trainer_name", ["GRPOTrainer", "SFTTrainer"])
def test_trainer_calls_only_use_arguments_the_trainer_has(trainer_name):
    import trl

    accepted = set(inspect.signature(getattr(trl, trainer_name).__init__).parameters)
    calls = calls_in_notebook(trainer_name)
    assert calls
    for _, call in calls:
        assert {kw.arg for kw in call.keywords} <= accepted


def notebook_config(config_name, tmp_path, **overrides):
    import trl

    ((_, call),) = calls_in_notebook(config_name)
    kwargs = keyword_values(call, {"OUTPUT_DIR": str(tmp_path), "USE_VLLM": False})
    kwargs.update(bf16=False, use_cpu=True, report_to="none", save_strategy="no", dataloader_num_workers=0,
                  logging_steps=1, **overrides)
    return getattr(trl, config_name)(**kwargs)


def logging_callback():
    ns = {"__name__": "__main__"}
    exec("import logging, sys\nfrom transformers import TrainingArguments, TrainerCallback, TrainerControl, "
         "TrainerState", ns)
    exec(cell_defining("LoggingCallback"), ns)
    return ns["LoggingCallback"]()


def test_two_grpo_steps_on_cpu_with_all_five_rewards(nb, tmp_path, caplog):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import GRPOTrainer

    ns = {}
    exec(cell_defining("make_conversation").replace("SYSTEM_PROMPT", repr("Think, then answer.")), ns)
    problems = [{"problem": f"What is {a} + {a}?", "solution": f"$\\boxed{{{2 * a}}}$"} for a in range(1, 9)]
    dataset = Dataset.from_list(problems).map(ns["make_conversation"])

    args = notebook_config("GRPOConfig", tmp_path, max_steps=2, per_device_train_batch_size=4,
                           gradient_accumulation_steps=1, num_generations=4, max_completion_length=16,
                           eval_strategy="no", eval_on_start=False)
    names = ["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty"]
    rewards = nb["get_reward_functions"](nb["GRPOScriptArguments"](reward_funcs=names))
    trainer = GRPOTrainer(model=AutoModelForCausalLM.from_pretrained(TINY_MODEL), reward_funcs=rewards,
                          args=args, train_dataset=dataset, callbacks=[logging_callback()])
    with caplog.at_level(logging.INFO, logger="__main__"):
        trainer.train()

    logged = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert len(logged) == 2
    for name in ("accuracy_reward", "format_reward", "reasoning_steps_reward", "cosine_scaled_reward",
                 "repetition_penalty_reward"):
        assert f"rewards/{name}/mean" in logged[-1]
    assert any(message.startswith("Step 2: Loss = ") and "format_reward = " in message
               for message in caplog.messages)


def test_two_sft_steps_on_cpu(tmp_path):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    conversations = [{"messages": [{"role": "system", "content": "Think step by step."},
                                   {"role": "user", "content": f"What is {a} + {a}?"},
                                   {"role": "assistant", "content": f"{a} + {a} = {2 * a}"}]} for a in range(8)]
    args = notebook_config("SFTConfig", tmp_path, max_steps=2, per_device_train_batch_size=2,
                           gradient_accumulation_steps=1, max_length=64)
    trainer = SFTTrainer(model=AutoModelForCausalLM.from_pretrained(TINY_MODEL),
                         train_dataset=Dataset.from_list(conversations),
                         processing_class=AutoTokenizer.from_pretrained(TINY_MODEL), args=args)
    trainer.train()
    assert len([entry for entry in trainer.state.log_history if "loss" in entry]) == 2


# ----------------------------------------------------------------------------- the notebook and the README

def test_the_notebook_was_run_from_top_to_bottom_without_errors():
    cells = notebook_cells()
    assert [c["execution_count"] for c in cells] == list(range(1, len(cells) + 1))
    for c in cells:
        assert not [out for out in c["outputs"] if out["output_type"] == "error"]


def readme_code_blocks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    return [normalize(block.split("#### OUTPUT ####")[0]) for block in blocks]


def normalize(code):
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def test_the_readme_shows_the_code_that_the_notebook_runs():
    blocks = readme_code_blocks()
    missing = [src.splitlines()[0] for src in code_sources() if normalize(src) not in blocks]
    assert not missing, missing


def test_requirements_are_pinned():
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            assert re.match(r"^[A-Za-z0-9_.\-\[\]]+==[0-9][^;\s]*(\s*;.*)?$", line), line
