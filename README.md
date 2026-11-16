![(title)](title.png)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)[![Medium](https://img.shields.io/badge/Medium-Read%20Now-red?logo=medium)](https://medium.com/@fareedkhandev) ![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow) 


The entire training process of DeepSeek R1 is nothing but using different way of reinforcement learning on top of their base model (i.e. [deepseek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3))

To make everything easy to understand we will use hand drawn flowcharts along with the code and will follow the step by step implementation of deepseek technical report and will build our own model using a tiny base model that you can also run locally.

We will also be covering theory next to each step, so in case you are not aware of DeepSeek R1 this blog will cover that too.

I explained **DeepSeek R1** with **hand-drawn diagrams** for non-tech folks. [Read here](https://levelup.gitconnected.com/drawing-deepseek-r1-architecture-and-training-process-from-scratch-72043da33955).

<!-- omit in toc -->
## GitHub Code Overview

The codebase is organized as follows:

```bash
train-deepseek-r1/
├── code.ipynb         # Jupyter notebook with code implementation
├── requirements.txt   # List of required libraries
├── r1_for_dummies.md  # DeepSeek R1 explanation for non-tech folks
├── tests/             # Tests for the notebook's reward functions and training setup
└── CHANGELOG.md       # What changed in each release
```

<!-- omit in toc -->
## Table of Contents
- [Setting up the Stage](#setting-up-the-stage)
- [Our Training Dataset](#our-training-dataset)
- [DeepSeek R1 Training Quick Overview](#deepseek-r1-training-quick-overview)
- [Choosing our Base Model](#choosing-our-base-model)
- [Policy Model (R) In RL Setup](#policy-model-r-in-rl-setup)
- [GRPO Algorithm for R1 Zero](#grpo-algorithm-for-r1-zero)
- [Prompt Template](#prompt-template)
- [Preprocessing Training Data](#preprocessing-training-data)
- [Reward Functions](#reward-functions)
  - [Accuracy Reward](#accuracy-reward)
  - [Format Reward](#format-reward)
  - [Reasoning Steps Reward](#reasoning-steps-reward)
  - [Cosine Scaled Reward](#cosine-scaled-reward)
  - [Repetition Penalty Reward](#repetition-penalty-reward)
- [Training Configurations for R1 Zero](#training-configurations-for-r1-zero)
- [GRPO Training Loop](#grpo-training-loop)
- [Saving Tiny R1 Zero LLM](#saving-tiny-r1-zero-llm)
- [Two main problems with R1 Zero](#two-main-problems-with-r1-zero)
- [Preparing Cold Start Data for SFT](#preparing-cold-start-data-for-sft)
- [Few-shot Prompting with Long CoT](#few-shot-prompting-with-long-cot)
- [Direct Prompting](#direct-prompting)
- [Post Processing Refinement](#post-processing-refinement)
- [SFT Stage 1 With Cold Start Data](#sft-stage-1-with-cold-start-data)
- [Stage 1 SFT Trainer Configs for R1](#stage-1-sft-trainer-configs-for-r1)
- [Stage 1 STF Training Loop](#stage-1-stf-training-loop)
- [Saving Tiny R1 LLM](#saving-tiny-r1-llm)
- [Reasoning-Oriented RL](#reasoning-oriented-rl)
- [Rejection Sampling](#rejection-sampling)
- [SFT Stage 2 Training](#sft-stage-2-training)
- [Distillation](#distillation)
- [Citation](#citation)



## Setting up the Stage

Clone the repository and install the required libraries using the following commands:

```bash
git clone https://github.com/FareedKhan-dev/train-deepseek-r1.git
cd train-deepseek-r1
pip install -r requirements.txt
```

You need Python 3.11 or newer. The notebook is tested with Python 3.12 on Linux with one NVIDIA GPU, and a few things are worth knowing before you install:

* The pinned PyTorch wheels are built for CUDA 13, which needs NVIDIA driver 580 or newer. With an older driver, `torch.cuda.is_available()` returns `False`.
* [vLLM](https://github.com/vllm-project/vllm) only installs on Linux. It makes the GRPO training about five times faster, but the notebook runs without it.
* On Windows, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) first: the Windows wheel on PyPI is CPU only.
* If you export the notebook to a script, call it something other than `code.py`, for example with `jupyter nbconvert --to script code.ipynb --output train_r1`. A `code.py` next to your script hides Python's built-in `code` module, which the debugger (`pdb`) imports.

Now, let’s import the required libraries and set up the environment for our training.

```python
# Import necessary libraries
import logging
import os
import sys
import re
import math
import importlib.util
from dataclasses import dataclass, field

# Import PyTorch and Hugging Face Transformers
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

# Import dataset utilities
from datasets import load_dataset

# Import libraries from TRL (Transformers Reinforcement Learning)
from trl import (
    GRPOTrainer,
    GRPOConfig,
    SFTTrainer,
    SFTConfig,
)

# Import math-related utilities
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
```

## Our Training Dataset

Although the paper doesn’t specify the exact initial dataset for RL pre-training, we assume it should be reasoning focused.

So to stay as close as possible to the original replication, we will use these two open-source reasoning Hugging Face datasets:

 1. [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) (For R1 Zero Training)

 2. [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) (For R1 Training)

AI-MO/NuminaMath-TIR contains 72,441 training and 99 test math problems, with a messages column showing the COT (chain of thought) reasoning behind the solution.

| Field    | Description |  
|----------|------------|  
| problem  | The math problem |  
| solution | Step-by-step solution |  
| messages    | Chat to solve the problem |

Take a look at its sample:
```python
# Load the "AI-MO/NuminaMath-TIR" dataset
MATH_le = load_dataset("AI-MO/NuminaMath-TIR", "default")  

# Access the first sample in the training set
MATH_le['train'][0]


#### OUTPUT ####
{
  'problem': 'What is the coefficient of $x^2y^6$ in the expansion of $\\left(\\frac{3 ...',
  'solution': 'To determine the coefficient of \\(x^2y^6\\) in the expansion of \\(\\left ...',
  'messages': [{'content': 'What is the coefficient of $x^2y^6$ in the expansion of $\\left(\\frac{3 ...', 'role': 'user'}, {'content': 'To determine the coefficient of \\(x^2y^6\\) in the expansion of \\(\\left ...', 'role': 'assistant'}],
}
#### OUTPUT ####
```

While Bespoke-Stratos contains 17K problems focused on math and code.

| Field        | Description |  
|-------------|------------|  
| system      | Guidelines for math and code problems |  
| conversation | Chat to solve the problem |

And its sample looks like:
```python
# Load the "Bespoke-Stratos-17k" dataset from bespokelabs
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default") 

# Access the first sample in the training set
bespoke_rl['train'][0]


#### OUTPUT ####
{
  'system': 'Your role as an assistant involves thoroughly exploring questions thro ...',
  'conversations': [{'from': 'user', 'value': 'Return your final response within \\boxed{}. The operation $\\otimes$ is ...'}, {'from': 'assistant', 'value': '<|begin_of_thought|>  Okay, let me try to figure out this problem. So, ...'}],
}
#### OUTPUT ####
```

It’s not necessary to choose these datasets only, you can select any of your choice as long as it is reasoning focused (**a problem and its step-by-step solution**).

## DeepSeek R1 Training Quick Overview

So, before going into the technical implementation, a quick overview is that DeepSeek-R1 isn’t trained from scratch, like, from nothing. Instead, they started with a pretty smart LLM they already had [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) but they wanted to make it a reasoning superstar.

![DeepSeek R1 Implementation Quick Overview (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5872/1*XhE5MVuPfOrrbBxgYtHMsg.png)

To do that, they used **Reinforcement Learning**, or RL for short where you reward the LLM when it does something good for reasoning while punish it otherwise.

But it’s not just one simple training session. It’s like a whole bunch of steps, a pipeline they call it. They first tried just pure **RL** to see if reasoning would pop up by itself **that was DeepSeek-R1-Zero**, kinda an experiment. Then for the **real DeepSeek-R1**, they made it more organized with different stages. They give it some starting data to get it going, then do RL, then more data, then more RL… it’s like leveling up, step by step!

The whole point is to make these language models way better at thinking through problems.
>So yeah, that’s the super short version before we look into the crazy details of each step

## Choosing our Base Model

Since DeepSeek team chose DeepSeek-V3 as their base model to create R1 Zero and R1, but it’s quite huge **685 GB 💀 in size** which is obviously not in our reach.

To keep it simple, we will use a much smaller base model [Qwen/Qwen2.5–0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (0.9 GB in size). If you have a higher GPU RAM that can even load unquantized LLMs, you can go for a bigger model, such as [Qwen/Qwen2.5–7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

Let’s take a look at some of the specification of our base model:

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-GRPO-training" # For saving our trained model

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer with chat template
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabulary size: {len(tokenizer)}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


#### OUTPUT ####
Vocabulary size: 151665
Model max length: 131072
Pad token: <|endoftext|>
EOS token: <|im_end|>
#### OUTPUT ####
```
These are some basic info about the model, take a look at the total number of parameters our base model has.

```python
# Initialize base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16
)

print(f"Model parameters: {model.num_parameters():,}")


#### OUTPUT ####
Model parameters: 494,032,768
#### OUTPUT ####
```

Close to 0.5B params, let’s print a simple response from it and then we will move on to next step.

```python
# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the appropriate device
model.to(device)

# Test basic inference
def test_model_inference(user_input: str):
    """Test basic model inference with the loaded model and tokenizer."""
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    # Decode only the newly generated tokens, not the prompt
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test the model
test_input = "how are you?"
response = test_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Model Response: {response}")


#### OUTPUT ####
Using device: cuda
Test Input: how are you?
Model Response: As an artificial intelligence language model, I don't have feelings or emotions like humans do, but I'm always here to assist you with any questions or tasks you may have! How can I help you today?
#### OUTPUT ####
```
So, the output of this tiny model is quite reliable and suitable for our DeepSeek lookalike model training for sure.

## Policy Model (R) In RL Setup

Now that we have selected our base model, next we need to understand how a basic RL setup works for training an LLM.

For DeepSeek R1 their starting point was (DeepSeek V3) base model and in our case we are starting with Qwen2.5–0.5B-Instruct. By a starting point I meant that **it has created the DeepSeek R1 zero version**, an initial version which has some errors in it before the final version was created.

The initial version (R1 Zero) was created using Reinforcement Learning where (DeepSeek v3/Qwen2.5–0.5B) acts as an RL agent (actor who takes action). Let’s first visualize how it works.

![Qwen 2.5 as an agent workflow (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5410/1*S6YIXu1vIVmQFl-DgRFktg.png)

The RL agent (DeepSeek V3/Qwen2.5-0.5B) starts by taking an **Action**, which means it generates an answer and some reasoning for a given problem that’s put into its **Environment**. The Environment, in this case, is simply the reasoning task itself.

After taking an action, the Environment gives back a **Reward**. This Reward is like feedback, it tells our base model (DeepSeek V3/Qwen2.5-0.5B) how good its action was. A positive Reward means it did something right, maybe got the answer correct or reasoned well. This feedback signal then goes back to our base model, helping it learn and adjust how it takes actions in the future to get even better Rewards.
> In the next section, we will be discussing this methodology in more detail

## GRPO Algorithm for R1 Zero

So that we have understand a basic RL flow now we need to learn what exact RL algorithm DeepSeek uses for R1-Zero.

There are many RL algos available, but traditional RL use something called a **“critic” **to help the main decision making part (“actor” i.e. DeepSeek-V3/Qwen2.5-0.5B). This critic is usually just as big and complex as the actor itself, which basically doubles the amount of computational cost.

But DeepSeek uses GRPO for training their initial (R1 Zero), **GRPO** does things differently because it figures out a baseline, a kind of reference point for good actions directly from the results it gets from a **group** of actions. Because of this, GRPO doesn’t need a separate critic model at all. This saves a lot of computation and makes things more efficient.

Let’s draw a flowchart of how GRPO is being used for R1 Zero training, and then we will **interpretate** it.

![GRPO Flow for DeepSeek R1 Zero (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6404/1*8mfNzi-gvasR7mSaseswmg.png)

Let’s understand how DeepSeek GRPO implementation works with our base model (Qwen2.5-0.5B). 

First, the **Problem Input (A)** is given to the **Qwen Model (B)**, Qwen attempts to generate an answer through **Generate Completion (C)**. The final result, called the **Completion Output (D)**, includes reasoning steps in <think> tags and the final solution in <answer> tags.

Next, the **Problem Input (A)** and the **Ground Truth Solution (E)** are fed into the **Reward Functions (F)**, acting as intelligent graders. These functions compare Qwen **Completion Output (D)** with the correct solution and evaluate different aspects such as:

 1. **Accuracy** (is the answer correct?)

 2. **Format** (are the <think> and <answer> tags used properly?)

 3. **Reasoning Steps** (is the logic clear?)

 4. **Cosine Scaling** (is the response concise?)

 5. **Repetition Penalty** (is there unnecessary repetition?).

These evaluations produce **Reward Scores (G)**, which are then passed to the **GRPO Trainer (H)**. The trainer uses gradients to adjust the **Qwen Model (B)**, fine-tuning how it generates answers. This process is called **Group Relative Policy Optimization** (GRPO): each completion's reward is compared with the rewards of the other completions for the same problem, and the model is pushed towards the completions that did better than their group.

Finally, the updated **Qwen Model (B)** is tested again on new problems, continuously refining itself through repeated cycles. With each iteration, Qwen becomes a better problem solver.

> In the upcoming section we will start preprocessing our training dataset for GRPO training

## Prompt Template

We are using the same thinking prompt template that DeepSeek uses for the GRPO algorithm to build R1 Zero, so let’s define that:

```python
# DeepSeek system prompt for GRPO based training
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
```
This **system prompt** tells the base model (Qwen2.5-0.5B) its role as a helpful assistant who reasons step-by-step before answering.

The `<think>` and `<answer>` tags are used to structure the model response, separating its internal reasoning from the final answer for better evaluation and reward.

## Preprocessing Training Data

Now that we have our system prompt ready, we need to transform our training data according to our template.

![Preprocessing dataset overview (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6160/1*XnM7v4dPD4LtyAh2MLuInA.png)

We need to create the make_conversation function that will handle the conversation for us.
```python
# Function to structure the training data
def make_conversation(example):
    """Convert dataset examples into conversation format."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }
```

It will take each problem column value from our training dataset and return a dictionary with the system prompt and the appended problem question for each row. Let’s create this function that will prepare our dataset.
```python
# Load and prepare dataset
def load_math_dataset():
    """Load and prepare the mathematics dataset."""
    dataset = load_dataset(
        "AI-MO/NuminaMath-TIR",
        name="default",
        split=['train', 'test']
    )
    
    # Convert splits into dictionary
    dataset = {
        'train': dataset[0],
        'test': dataset[1]
    }
    
    # Apply conversation format
    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)

        # Remove 'messages' column if exists
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    return dataset
```
We have everything ready, let’s transform our training data into the required format and print the training and test size.

```python
# Load our training dataset and printing train/test size
dataset = load_math_dataset()

print(f"Train set size: {len(dataset['train'])}")
print(f"Test set size: {len(dataset['test'])}")


#### OUTPUT ####
Train set size: 72441
Test set size: 99
#### OUTPUT ####
```
Now that we have split our training dataset, we need to validate our dataset (**Check if user/assistant conversation exist**) before moving to the next step.
```python
def validate_dataset(dataset):
    """Perform basic validation checks on the dataset."""
    
    # Define the required fields for the dataset
    required_fields = ["problem", "prompt"]

    # Loop through the 'train' and 'test' splits of the dataset
    for split in ['train', 'test']:
        print(f"\nValidating {split} split:")

        # Retrieve column names from the dataset
        fields = dataset[split].column_names

        # Check if any required fields are missing
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"Warning: Missing fields: {missing}")  # Warn if fields are missing
        else:
            print("✓ All required fields present")  # Confirm all fields are present

        # Retrieve the first sample from the dataset split
        sample = dataset[split][0]

        # Extract the 'prompt' field, which contains a list of messages
        messages = sample['prompt']

        # Validate the prompt format:
        # - It should contain at least two messages
        # - The first message should be from the 'system' role
        # - The second message should be from the 'user' role
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ Prompt format is correct")  # Confirm correct format
        else:
            print("Warning: Incorrect prompt format")  # Warn if format is incorrect

# Validate dataset
validate_dataset(dataset)
```

It outputs this:

```
Validating train split:
✓ All required fields present
✓ Prompt format is correct

Validating test split:
✓ All required fields present
✓ Prompt format is correct
```
Our training dataset is validated successfully 🙌, it means we have successfully transformed our dataset for training.

## Reward Functions

We already saw in GRPO section that it evaluate the answer of base model through five different ways:

![Reward Functions (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7474/1*kJln8i6Tv4aspnTfMoRW-Q.png)

 1. **Accuracy** (is the answer correct?)

 2. **Format** (are the `<think>` and `<answer>` tags used properly?)

 3. **Reasoning Steps** (is the logic clear?)

 4. **Cosine Scaling** (is the response concise?)

 5. **Repetition Penalty** (is there unnecessary repetition?).

Each of these are functions will calculate the reward for each response, and we need to code them. So, let’s do that first.

### Accuracy Reward

Accuracy reward is the most easy to understand but requires a bit complex code. In this reward model we want to check if mathematically our base model response is equivalent to the ground truth solution.

![Accuracy Reward (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7860/1*A3tW-OZSZ4m10EEzogjy8Q.png)

If the model answer is mathematically correct, we assign a reward of **1.0**. If it is incorrect, the reward is **0.0**. In cases where the ground truth solution cannot be parsed, we assign a neutral reward of **0.5** to avoid unfair penalties.

Now, let’s implement the function.
```python
def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function to check if the model's response is mathematically 
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """
    
    # Extract responses
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Parse the ground truth solution
        gold_parsed = parse(sol, extraction_mode="first_match", 
                            extraction_config=[LatexExtractionConfig()])
        
        if gold_parsed:  # Check if parsing was successful
            # Parse the model's answer with relaxed normalization
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # Reward 1.0 if correct, 0.0 if incorrect (the gold answer goes first, verify() is not symmetric)
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If ground truth cannot be parsed, assign neutral reward (0.5)
            reward = 0.5
            print("Warning: Failed to parse gold solution:", sol)

        rewards.append(reward)
    
    return rewards
```
In this function, we check whether the model response is **equivalent** to the correct answer. Instead of comparing raw text, we:

 1. Convert the solution into a structured mathematical format using **latex2sympy2**.

 2. If parsing fails, assign a neutral reward of **0.5**.

 3. Extract the model output and normalize it for better robustness.

 4. Use **math_verify** to check if the parsed response matches the parsed solution.

 5. If correct assign **1,** if incorrect assign **0**.

This ensures that accuracy evaluation is not just about textual similarity but **true mathematical correctness.**

### Format Reward

Format Reward is all about making sure our model follows instructions and structures its output correctly. We asked it to put its reasoning in `<think>` tags and the final answer in `<answer>` tags, right? This reward function checks exactly that!

![Forward Reward (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6620/1*DbUraziwiOoAj6SvtSJmpw.png)

If the model uses those tags correctly, we give it a reward of 1. If it messes up the format, it gets 0. Simple as that! This encourages the model to pay attention to the output structure we want.

Let’s code this up:
```python
# Implement Format Reward Function
def format_reward(completions, **kwargs):
    """
    Reward function to check if the completion has the correct format:
    <think>...</think> <answer>...</answer>.
    """
    # Define the regex pattern for the desired format
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    # Extract the content from each completion
    completion_contents = [completion[0]["content"] for completion in completions]

    # Check if each completion matches the pattern
    matches = [re.match(pattern, content, re.DOTALL)
               for content in completion_contents]

    # Reward 1.0 for correct format, 0.0 otherwise
    return [1.0 if match else 0.0 for match in matches]
```

In this function:

* We define a pattern using regular expressions (regex). This pattern basically says “the content should *start* with <think>, have *anything* inside until </think>, then some *spaces*, then <answer>, *anything* inside until </answer>, and then *end* there”.

* We get the actual text content from each model completion.

* Then we use re.match to see if each content perfectly matches our pattern. re.DOTALL lets the . in the regex match newlines too. We do not pass re.MULTILINE: with it, $ also matches at the end of every line, so a completion that keeps writing after `</answer>` would still get the full reward.

* Finally, we give a reward 1 if it matched the format perfectly, 0 if it didn’t. This is a strict on/off reward for format correctness.

### Reasoning Steps Reward

Reasoning Steps Reward is a bit clever. We want to encourage our model to show its **“thinking process”**. So, we are going to reward it for including things that *look like* reasoning steps.

![Reasoning Steps Reward Encouragement (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5406/1*hx0sAVnY58WOYw6rGF64ug.png)

We will look for keywords and patterns that usually show up in step-by-step reasoning, like:

* Step 1, Step 2, etc.

* Numbered lists like 1, 2

* Bullet points like - or *

* Transition words like First, Second, Next, Finally

The more of these it includes, the better the reward. It’s like giving points for showing its work!

Let’s code this reasoning encouraging function:
```python
def reasoning_steps_reward(completions, **kwargs):
    r"""
    Reward function to encourage clear step-by-step reasoning.
    It looks for patterns like "Step 1:", numbered lists, bullet points,
    and transition words.
    """
    # Regex pattern to find indicators of reasoning steps
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Count the number of reasoning step indicators in each completion
    matches = [len(re.findall(pattern, content, re.MULTILINE))
               for content in completion_contents]

    # Reward is proportional to the number of reasoning steps, maxing out at 1.0
    # We're using a "magic number" 3 here - encourage at least 3 steps for full reward
    return [min(1.0, count / 3) for count in matches]
```

We create a pattern that’s a bit more complex regex. It looks for all those reasoning indicator things we listed above.

We use re.findall to find *all* the matches of our pattern within each content. `len(re.findall(…))` then gives us the *count* of these indicators.

The reward is calculated as min(1.0, count / 3). This means

* If it finds 3 or more reasoning indicators ( count >= 3), the reward is 1.0 (max reward).

* If it finds fewer (e.g., count = 1 or 2), it gets a *partial* reward (like 1/3 or 2/3).

* If it finds none (count = 0), the reward is 0.0.

The / 3 is a bit of a magic number here. We’re saying **“aim for about 3 reasoning steps to get full credit”** You can tweak this number if you want to encourage more or fewer steps.

### Cosine Scaled Reward

Cosine Scaled Reward is a bit more advanced. It’s about encouraging *conciseness* in correct answers and being *less harsh* on longer incorrect answers.

![Cosine Scaling Concept (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7094/1*WmG8r1OVeU4R3jObAy0yCg.png)

Think of it like this:

* **For correct answers:** We want to reward *shorter*, more direct solutions more than long, rambling ones. A short, correct answer is often better.

* **For incorrect answers:** A short, wrong answer is probably worse than a longer, wrong answer that at least *tried* to reason. So, we want to penalize short wrong answers *more* than long wrong answers.

Let’s see the code that does this clever scaling:
```python
# Implement Cosine Scaled Reward Function
def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """
    def cosine_scaled_reward(completions, solution, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """
        contents = [completion[0]["content"] for completion in completions]
        accuracy_rewards = accuracy_reward(completions, solution) # 1.0 correct, 0.0 wrong, 0.5 unparseable gold
        rewards = []

        for content, acc_reward in zip(contents, accuracy_rewards):
            if acc_reward == 0.5: # The gold answer could not be parsed, so there is nothing to scale
                rewards.append(0.0)
                continue

            gen_len = len(content)  # Length of the generated answer (in characters)
            progress = min(gen_len / max_len, 1.0) # How far we are to max length (capped at 1)
            cosine = math.cos(progress * math.pi) # Cosine value based on progress

            if acc_reward > 0.5: # Assuming accuracy_reward gives ~1.0 for correct answers
                min_value = min_value_correct
                max_value = max_value_correct
            else: # Incorrect answer
                min_value = max_value_wrong  # Note the swap!
                max_value = min_value_wrong

            # Cosine scaling formula!
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward
```

`get_cosine_scaled_reward(...)` generates a reward function for training, customizing scaling with parameters like min_value_wrong/max_value_wrong (penalty range for incorrect answers) and min_value_correct/max_value_correct (reward range for correct ones). max_len sets the maximum length for scaling.

Inside, `cosine_scaled_reward(...)` first scores every completion with our `accuracy_reward`, so it only needs the completions and the solution column, which the trainer passes to every reward function. If the gold answer cannot be parsed (accuracy 0.5), the completion gets 0.

It computes gen_len (the length of the completion in characters), normalizes it as progress `= min(gen_len / max_len, 1)` and derives a cosine value that starts at 1 (short answers) and decreases to -1 (answers of max_len characters or longer). Without the cap, the cosine would climb back up for completions longer than max_len, and a 2,000 character answer would score like an empty one.

If `acc_reward > 0.5`, it uses the correct reward range, otherwise it applies the incorrect range but swaps min/max values to penalize longer wrong answers less.

### Repetition Penalty Reward

Repetition Penalty Reward is all about discouraging our model from getting stuck in loops and repeating itself. We want it to generate fresh, varied reasoning and answers, not just copy-paste the same phrases over and over!

![Repetition Penalty Idea (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/8608/1*9jBhiz-rI_fRGa77g9RZtQ.png)

This reward function penalizes the model if it uses the same sequences of words (n-grams) too many times. We’ll use n-grams of size 3 (trigrams) in our example, but you can adjust this.

If the model repeats itself a lot, it gets a negative reward (penalty). If it’s more diverse and avoids repetition, the penalty is less.

Let’s implement the code to penalize repetition:
```python
def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """Helper function to generate n-grams from text."""
        words = text.lower().split() # Lowercase and split into words
        return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "": # No penalty for empty completions
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size: # No penalty for short completions
                rewards.append(0.0)
                continue

            ngrams = set() # Use a set to store unique n-grams
            total = 0
            for ng in zipngram(completion, ngram_size): # Generate n-grams
                ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
                total += 1 # Count total n-grams

            # Calculate scaling factor: more repetition -> higher scaling
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty # Apply penalty based on scaling
            rewards.append(reward)
        return rewards
    return repetition_penalty_reward
```

Our `get_repetition_penalty_reward(...)` creates a reward function to penalize repetition, with parameters like ngram_size (default 3, for trigrams) and max_penalty (a negative value, e.g., -0.1).

A helper function, `zipngram(text, ngram_size)`, generates n-grams by converting text to lowercase, splitting it into words, and using `zip(*[words[i:] for i in range(ngram_size)])` for efficient extraction.

Inside, `repetition_penalty_reward(...)` computes the penalty for each completion. If it's empty or too short, it gets a reward of 0.0.

The penalty scales as scaling `= 1 - len(ngrams) / total`, where total is the number of n-grams and len(ngrams) is the unique count. More repetition makes scaling approach 1, increasing the penalty.

The final reward is scaling * max_penalty, meaning less repetition results in a smaller penalty, while high repetition leads to a stronger negative reward. 

>We have implemented all five reward functions, Let’s move on to next stage where we define our training args

## Training Configurations for R1 Zero

Now we to code a configuration where we can fine-tune how our *reward functions* actually work. So, Let’s define that configuration class:

```python
# Define GRPOScriptArguments for reward function parameters
@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward for cosine scaling for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for cosine scaling for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward for cosine scaling for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for cosine scaling for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for cosine scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
```

Our `@dataclass` decorator makes it easy to create a class for storing data. WhileGRPOScriptArguments class holds reward settings.

The reward_funcs list decides which rewards to use, starting with ["accuracy", "format"], but you can add more like "reasoning_steps", "cosine", "repetition_penalty".

Some settings control how the cosine_scaled_reward and repetition_penalty_reward work, letting you adjust how rewards are given.

Next up, we have GRPOConfig from TRL. It extends the TrainingArguments class of transformers, so it is the **main** configuration object that controls almost **everything** about the training process, and it adds the GRPO settings, such as how many completions to sample for each problem.
```python
# Use vLLM to generate the completions when it is installed (Linux with an NVIDIA GPU).
# It is several times faster than model.generate(), which the trainer uses otherwise.
USE_VLLM = torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None

# Define the training configuration (GRPOConfig extends TrainingArguments from transformers)
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,          # Output directory for checkpoints and logs
    num_train_epochs=1,             # Total number of training epochs
    max_steps=300,                  # Stop after 300 steps (remove it to train the whole epoch)
    per_device_train_batch_size=8,  # Completions per device per step (one problem x 8 generations)
    per_device_eval_batch_size=16,  # Batch size for evaluation
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    learning_rate=1e-6,             # Initial learning rate for AdamW optimizer
    warmup_steps=0.1,               # Linear warmup over the first 10% of the training steps
    weight_decay=0.01,              # Apply weight decay to all layers except bias and LayerNorm weights
    logging_steps=10,               # Log every X updates steps
    eval_strategy="steps",          # Evaluate every `eval_steps`
    eval_steps=50,                  # Evaluation and logging steps
    eval_on_start=True,             # Also evaluate before training, to compare with
    save_strategy="steps",          # Save checkpoint every `save_steps`
    save_steps=50,                  # Save checkpoint every X updates steps
    save_total_limit=2,             # Limit the total amount of checkpoints. Deletes the older checkpoints.
    dataloader_num_workers=2,       # Number of subprocesses to use for data loading
    seed=42,                        # Random seed for reproducibility
    bf16=True,                      # Use BF16 (bfloat16) precision
    push_to_hub=False,              # Whether to push the final model to Hugging Face Hub
    gradient_checkpointing=True,    # Enable gradient checkpointing
    report_to="none",               # Reporting to no one
    remove_unused_columns=False,    # Keep the "solution" column, the accuracy reward needs it
    # GRPO settings
    num_generations=8,              # Completions sampled for each problem (the "group" in GRPO)
    max_completion_length=512,      # Longest completion in tokens, longer ones are cut off
    use_vllm=USE_VLLM,              # Generate with vLLM when it is available
    vllm_mode="colocate",           # Run vLLM inside this process, on the same GPU
    vllm_gpu_memory_utilization=0.3,  # Share of the GPU memory that vLLM may use
)
```

A few of these settings matter a lot for GRPO:

* **num_generations**: GRPO samples a group of completions for every problem and compares their rewards with each other. With 8 completions per problem and 16 completions per optimizer step (8 per device times 2 accumulation steps), each step learns from 2 problems.

* **max_completion_length**: completions are cut off at 512 tokens. A completion that is cut off never reaches `</answer>`, so it gets no format reward.

* **max_steps**: one epoch over the 72,441 training problems is 36,220 steps, which takes about a day on one RTX A6000 with vLLM. We stop after 300 steps, enough to see how training behaves. Remove `max_steps` to train the whole epoch.

* **learning_rate**: RL fine-tuning needs a much smaller learning rate than supervised fine-tuning. We use 1e-6, TRL's default for GRPO. With 5e-5, the value earlier versions of this notebook used, the accuracy on the 99 test problems fell from 16.6% to 0% within 100 steps and stayed below 1%: every completion got the same reward, so there was nothing left to learn from.

* **use_vllm**: generating the completions is the slowest part of GRPO. [vLLM](https://github.com/vllm-project/vllm) does it several times faster than `model.generate()`, so we use it when it is installed (it needs Linux and an NVIDIA GPU). Everything else works without it, only slower.

Now we need to actually **create** an instance of our reward settings:
```python
# Instantiate configuration objects
script_args = GRPOScriptArguments()
```

Next, we need to get our list of reward functions and any “callbacks” we want to use during training.

Callbacks are like little helpers that can do things at different points in the training process (like logging progress, saving models, etc.). For now, we’ll just use a simple logging callback.

Getting our reward functions in one place.
```python
# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,  # Assuming accuracy_reward is defined in previous steps
        "format": format_reward,      # Assuming format_reward is defined in previous steps
        "reasoning_steps": reasoning_steps_reward, # Assuming reasoning_steps_reward is defined
        "cosine": get_cosine_scaled_reward( # Assuming get_cosine_scaled_reward is defined
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward( # Assuming get_repetition_penalty_reward is defined
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list
```
Our callback function which will track loss and other important info. It runs every time the trainer logs, so it reads the numbers the trainer has just computed.
```python
logging.basicConfig(stream=sys.stdout)  # Print log messages in the notebook
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Show our INFO messages (other libraries stay at WARNING)

class LoggingCallback(TrainerCallback):
    """
    A simple callback that logs the loss, the learning rate and the mean of each reward
    every time the trainer logs (every `logging_steps` steps).
    """
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs or "loss" not in logs:  # Skip evaluation logs, they carry "eval_loss" instead
            return
        message = f"Step {state.global_step}: Loss = {logs['loss']:.4f}, Learning Rate = {logs.get('learning_rate', 0):.2e}"
        for key, value in logs.items():
            if key.startswith("rewards/") and key.endswith("/mean"):  # e.g. rewards/format_reward/mean
                message += f", {key.split('/')[1]} = {value:.3f}"
        logger.info(message)

def get_callbacks(training_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()] # Instantiate our LoggingCallback
    return callbacks
```

Finally, initializing these function.
```python
# Get reward functions and callbacks
reward_functions = get_reward_functions(script_args)
callbacks = get_callbacks(training_args, script_args)
```

## GRPO Training Loop

This is the engine that will actually drive our GRPO training. We need to initialize it, giving it all the pieces we’ve prepared: our model, reward functions, training arguments, dataset, and callbacks!

Let’s initialize the GRPOTrainer:
```python
# Initialize the GRPO trainer
grpo_trainer = GRPOTrainer(
    model=model,                      # Our initialized Qwen model
    reward_funcs=reward_functions,    # List of reward functions from previous step
    args=training_args,               # GRPOConfig defined above
    train_dataset=dataset['train'],   # Training dataset
    eval_dataset=dataset['test'],     # Evaluation dataset
    callbacks=callbacks               # List of callbacks
)
```

We can now start the **Training Loop**! This is as simple as calling the train() method on our grpo_trainer.
```python
# Start the GRPO Training Loop
train_result = grpo_trainer.train()
```
When you run this cell, the training starts, and our callback prints a line every 10 steps:
```
INFO:__main__:Step 10: Loss = 0.0137, Learning Rate = 3.00e-07, accuracy_reward = 0.113, format_reward = 0.000
INFO:__main__:Step 20: Loss = -0.0026, Learning Rate = 6.33e-07, accuracy_reward = 0.094, format_reward = 0.000
INFO:__main__:Step 30: Loss = 0.0219, Learning Rate = 9.67e-07, accuracy_reward = 0.087, format_reward = 0.000
INFO:__main__:Step 40: Loss = -0.0401, Learning Rate = 9.67e-07, accuracy_reward = 0.250, format_reward = 0.000
INFO:__main__:Step 50: Loss = 0.0209, Learning Rate = 9.30e-07, accuracy_reward = 0.081, format_reward = 0.000
...
INFO:__main__:Step 100: Loss = -0.0225, Learning Rate = 7.44e-07, accuracy_reward = 0.113, format_reward = 0.000
...
INFO:__main__:Step 150: Loss = -0.0235, Learning Rate = 5.59e-07, accuracy_reward = 0.044, format_reward = 0.000
...
INFO:__main__:Step 200: Loss = -0.0019, Learning Rate = 3.74e-07, accuracy_reward = 0.094, format_reward = 0.000
...
INFO:__main__:Step 250: Loss = 0.0269, Learning Rate = 1.89e-07, accuracy_reward = 0.131, format_reward = 0.000
...
INFO:__main__:Step 300: Loss = 0.0212, Learning Rate = 3.70e-09, accuracy_reward = 0.125, format_reward = 0.000
```
On one RTX A6000 with vLLM, the 300 steps took 23 minutes: 2.6 seconds per step, plus about 95 seconds for each of the seven evaluations on the 99 test problems. Without vLLM a step takes about 13 seconds. The GPU memory peaked at 20 GB, most of it the 30% share we gave vLLM; without vLLM the peak is 5.7 GB.

The evaluations show what the 300 steps did:

| Step | Accuracy reward | Format reward | Mean completion length |
|---:|---:|---:|---:|
| 0 (before training) | 0.166 | 0.000 | 413 tokens |
| 50 | 0.158 | 0.001 | 416 tokens |
| 100 | 0.147 | 0.000 | 411 tokens |
| 150 | 0.154 | 0.001 | 416 tokens |
| 200 | 0.151 | 0.000 | 415 tokens |
| 250 | 0.160 | 0.000 | 417 tokens |
| 300 | 0.168 | 0.000 | 417 tokens |

Not much changed, and that is what to expect: 300 steps of 2 problems each, at a learning rate this small, barely move a model. The accuracy went from 16.6% to 16.8%. This run shows the mechanics; the whole epoch is 36,220 steps.

The format reward stays at about 0 for a different reason: the base model almost never writes the `<think>` and `<answer>` tags. In 96 completions we sampled with this prompt, one contained `<think>` and none matched the format. When every completion of a problem gets the same reward, GRPO has nothing to compare, so the format reward cannot teach anything until the model writes the format now and then. DeepSeek-R1 avoids this cold start of RL by first fine-tuning on a small amount of long chain-of-thought data, which we get to below.

With the learning rate of 5e-5 that earlier versions of this notebook used, the run broke down instead: the accuracy on the test problems fell from 16.6% to 0% within 100 steps and stayed below 1%.

But for real-world GRPO DeepSeek R1 Zero training, you’d likely train for many more epochs and steps.

## Saving Tiny R1 Zero LLM

Once the training completed, we can save our trained model which can be used for inferencing.
```python
# Define the path to your trained model (same as OUTPUT_DIR)
TRAINED_MODEL_PATH = "data/Qwen-GRPO-training"

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

# Save the trained model
grpo_trainer.save_model(TRAINED_MODEL_PATH)

print(f"GRPO Trained model saved to {TRAINED_MODEL_PATH}")


#### OUTPUT ####
GRPO Trained model saved to data/Qwen-GRPO-training
#### OUTPUT ####
```
Then we can simply load the trained model using:
```python
# Load the tokenizer - make sure to use trust_remote_code=True if needed
tokenizer = AutoTokenizer.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model config requires it
    padding_side="right" # Ensure consistent padding side
)

# Set pad token if it wasn't saved or loaded correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the trained model itself
trained_model = AutoModelForCausalLM.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model architecture requires it
    dtype=torch.bfloat16 # Keep the same dtype as training for consistency
)

# Move the loaded model to your device (GPU if available)
trained_model.to(device) # 'device' is still our CUDA device from before
```

In order to use it for inference:

```python
# Testing Inference with the Trained Model
def test_trained_model_inference(user_input: str):
    """Test inference with the loaded trained model and tokenizer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # Re-use our system prompt
        {"role": "user", "content": user_input}
    ]

    # Apply chat template using our tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate output using our *trained_model*
    outputs = trained_model.generate(
        **inputs,
        max_new_tokens=512, # Same limit as max_completion_length in training
        do_sample=True,
        temperature=0.7
    )

    # Decode the generated tokens back to text (only the new ones, not the prompt)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test the model on a problem from the test split
test_input = dataset['test'][0]['problem']
response = test_trained_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Trained Model Response: {response}")


#### OUTPUT ####
Test Input: In 1988, a person's age was equal to the sum of the digits of their birth year. How old was this person?
Trained Model Response: To solve this problem, we need to find a year such that when its digits are added together, the result is equal to the original year.

Let's start by listing the possible years:

- 20th century: 2000, 21st century: 2100, etc.

We will check each year to see if adding the digits results in the same number as the original year.

For example:
- For 2000: \( 2 + 0 + 0 = 2 \)
- For 2100: \( 2 + 1 + 0 = 3 \)
...
The only year that fits the criteria is 2016. Therefore, the person's age in 1988 was 2016.

So, the answer is 2016.
#### OUTPUT ####
```

The response is sampled (`do_sample=True`), so yours will be different. Like the evaluations above, it shows a model that still answers in its usual style, without the `<think>` and `<answer>` tags, and here it gets the answer wrong (the person was 22).

Before we move on, we free the GPU memory that the GRPO trainer still holds:
```python
# Free the GPU memory used by the GRPO stage
import gc
del grpo_trainer, trained_model, model
gc.collect()
torch.cuda.empty_cache()
```

## Two main problems with R1 Zero

Now that we have completed our R1 zero training approach using our base model Qwen2.5-0.5Binstead of their DeepSeek V3 (original base model).

We cannot identify our trained model problems but researches of DeepSeek saw the R1 Zero model performed really well on reasoning tests, even scoring similarly to more advanced models like **OpenAI-o1-0912**on tasks like **AIME 2024**.

This showed that using reinforcement learning (RL) to encourage reasoning in language models is a promising approach.

But they also noticed DeepSeek-R1-Zero had some key issues that needed fixing for real world use and wider research.

![Problem with R1 Zero (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6378/1*_NdVhpb9cgT3-8o3Qn7mMA.png)

Researchers of DeepSeek states that the template is *intentionally simple and structurally focused*. It *avoids* imposing any *content-specific* constraints on the *reasoning process itself*. For example, it doesn’t say:

* “You *must* use step-by-step reasoning” (It just says “reasoning process” leaving it open to the model to define what that means).

* “You *must* use reflective reasoning”

* “You *must* use a specific problem-solving strategy”

The main problem was that the reasoning processes inside the `<think>` tags were hard to read, making it tough for humans to follow and analyze.

Another issue was language mixing, when asked multi-lingual questions, the model sometimes mixed languages in the same response, leading to inconsistent and confusing outputs.

If you asked it questions in, say, Spanish. Suddenly, its “thinking” would be a jumbled mix of **English and Spanish, **not exactly polished! These problems, messy reasoning and language confusion, were the clear roadblocks.
> These are the two main reasons they transformed their initial R1 Zero Model into the R1

## Preparing Cold Start Data for SFT

So to fix R1 Zero issues and really get DeepSeek reasoning properly, researchers performed a **Cold Start Data Collection and included Supervised Fine Tuning**.

You can think of it as giving the model a good foundation in reasoning before the really intense RL training. Basically, they wanted to teach **DeepSeek-V3 Base** what good reasoning looks like and how to present it clearly.

One of the example of cold start data is [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) that we see earlier and will be using for creating R1, but **we need to understand how cold dataset is created so we wont skip any part from the actual training**.

## Few-shot Prompting with Long CoT

One technique is **Few-shot Prompting with Long Chain-of-Thought (CoT),** in which we try to show DeepSeek-V3 Base (or in our case, Qwen2.5-0.5B) few examples of questions paired with super detailed, step-by-step solutions. This is Chain-of-Thought (CoT).

![Long CoT (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/4068/1*SAhvB0JqaK4d45IiIcj1Ow.png)

Goal of this approach is to make the model learn by example and start mimicking this thorough reasoning style.

For our example problem “What is 2 + 3 * 4?”, we can create prompts that include a few solved problems as examples. Let’s see how this looks in Python:
```python
# Loading Model and Tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate Long COT Response
def generate_response(prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides step-by-step solutions."},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False) # Keep it deterministic for example
    # Decode only the new tokens, which are the assistant's response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()
```
Let’s define the few shot examples accordingly for our asked question:
```python
# Example problems with solutions (using | special_token | as delimiter)
few_shot_prompt = """
Problem: What's the square root of 9 plus 5?
Solution: <|special_token|> First, find the square root of 9, which is 3. Then, add 5 to 3.  3 + 5 equals 8. <|special_token|> Summary: The answer is 8.

Problem: Train travels at 60 mph for 2 hours, how far?
Solution: <|special_token|> Use the formula: Distance = Speed times Time. Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. <|special_token|> Summary: Train travels 120 miles.
"""
```

Now using our base model our sample generations looks like this:
```python
# Generate response for the target problem using few-shot examples
target_problem_prompt = few_shot_prompt + "\nProblem: What is 2 + 3 * 4?\nSolution:"
model_response_few_shot = generate_response(target_problem_prompt)

print("Few-shot Prompt:")
print(target_problem_prompt)
print("\nModel Response (Few-shot CoT):")
print(model_response_few_shot)
```

It outputs this:

```
Few-shot Prompt:

Problem: What's the square root of 9 plus 5?
Solution: <|special_token|> First, find the square root of 9, which is 3. Then, add 5 to 3.  3 + 5 equals 8. <|special_token|> Summary: The answer is 8.

Problem: Train travels at 60 mph for 2 hours, how far?
Solution: <|special_token|> Use the formula: Distance = Speed times Time. Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. <|special_token|> Summary: Train travels 120 miles.

Problem: What is 2 + 3 * 4?
Solution:

Model Response (Few-shot CoT):
The problem "What is 2 + 3 * 4?" requires us to follow the order of operations (PEMDAS/BODMAS), which stands for Parentheses/Brackets, Exponents/Orders, Multiplication and Division (from left to right), Addition and Subtraction (from left to right).

Here’s the step-by-step solution:

1. **Parentheses**: There are no parentheses in this expression.
2. **Exponents**: There are no exponents in this expression.
3. **Multiplication and Division**: Perform these operations from left to right:
   - First, multiply 3 by 4: \(3 \times 4 = 12\).
   - Next, perform addition: \(2 + 12 = 14\).

So, the final result is \(14\).
```

The model reasons step by step and gets 14, but it answers in its usual markdown style instead of copying the `<|special_token|>` layout of the examples: two examples were not enough for this small model to pick up the format.

## Direct Prompting

Another method is **Direct Prompting**. Here, we directly instruct the model to not just solve the problem, but also to explicitly show its reasoning step-by-step and then verify its answer. 

This is about encouraging a more deliberate and thoughtful problem-solving approach.

![Example based learning (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/4656/1*IYyk7UWgDNADFe_djWcXow.png)

Let’s craft a prompt for “What is 2 + 3 * 4?” that explicitly asks for reasoning and verification. Here’s the Python code to see it in action:
```python
# Direct prompting example
direct_prompt_text = """
Problem: Solve this, show reasoning step-by-step, and verify:
What is 2 + 3 * 4?
"""

model_response_direct = generate_response(direct_prompt_text)

print("Direct Prompt:")
print(direct_prompt_text)
print("\nModel Response (Direct Prompting):")
print(model_response_direct)
```
The direct prompting output looks like this:
```
Direct Prompt:

Problem: Solve this, show reasoning step-by-step, and verify:
What is 2 + 3 * 4?


Model Response (Direct Prompting):
To solve the expression \(2 + 3 \times 4\), we need to follow the order of operations, often remembered by the acronym PEMDAS (Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right)).

Step 1: Evaluate the multiplication first.
\[3 \times 4 = 12\]

So now our expression looks like this:
\[2 + 12\]

Step 2: Perform the addition.
\[2 + 12 = 14\]

Therefore, the result of \(2 + 3 \times 4\) is \(14\).

**Verification:**
Let's break it down step-by-step:

1. **Multiplication**: First, perform the multiplication inside the parentheses.
   \[3 \times 4 = 12\]
   
2. **Addition**: Now add the results from the previous steps.
   \[2
```
By directly asking for reasoning and verification, we get a longer answer that ends with a **Verification** section (cut off mid-sentence, because `generate_response` allows 200 new tokens).

This method directly guides the model to produce the kind of detailed reasoning we are looking for.

## Post Processing Refinement

The final technique involves **Post-Processing Refinement**. Interestingly, they even used the outputs from the already trained R1 Zero model for this!

Even with its issues, R1 Zero could reason somewhat. So, they took R1 Zero outputs and had human annotators refine them, making them cleaner, more structured, and correcting any mistakes.

![Processing Refnement (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/4388/1*-GR29EAnTOVBarQ2JrF5sA.png)

Imagine a messy R1 Zero output like this:
```
<think>  ummm... multiply 3 and 4... get 12... then add 2...</think>
<answer> 14 </answer>
```

Human annotators would then refine it to something much clearer and better formatted:
```
<|special_token|> Reasoning: To solve this, we use order of operations, doing multiplication before addition.
Step 1: Multiply 3 by 4, which is 12.
Step 2: Add 2 to the result: 2 + 12 = 14.
<|special_token|> Summary: The answer is 14.
```

While we can’t perfectly simulate human refinement in code, we can demonstrate a basic idea of how you might programmatically reformat and structure a potentially messy output. 

Let’s take a simulated “messy” output and show how we could refine it:
```python
# Simulated messy R1 Zero output
messy_output = "<think>  ummm... multiply 3 and 4... get 12... then add 2...</think>\n<answer> 14 </answer>"

def refine_output(messy_text):
    think_content = messy_text.split("<think>")[1].split("</think>")[0].strip()
    answer_content = messy_text.split("<answer>")[1].split("</answer>")[0].strip()

    # Drop the filler word at the start ("umm...", "ummm...") and the dots at the end
    think_content = re.sub(r"^u+m+\.*\s*", "", think_content).rstrip(".")
    refined_text = f"""<|special_token|> Reasoning: {think_content.capitalize()}.
<|special_token|> Summary: The answer is {answer_content}."""
    return refined_text

refined_output_text = refine_output(messy_output)

print("Messy Output (Simulated R1 Zero):")
print(messy_output)
print("\nRefined Output:")
print(refined_output_text)
```

This will output:
```
Messy Output (Simulated R1 Zero):
<think>  ummm... multiply 3 and 4... get 12... then add 2...</think>
<answer> 14 </answer>

Refined Output:
<|special_token|> Reasoning: Multiply 3 and 4... get 12... then add 2.
<|special_token|> Summary: The answer is 14.
```

This simple refine_output function is just a basic example. Real refinement by humans involves much more nuanced understanding and correction of reasoning steps.

However, it shows the core idea: taking initial model outputs and improving their quality and structure to create better training data.
> After generating this Cold Start Data, the next crucial step was **Supervised Fine-Tuning (SFT)**, which we’ll explore in the next section!

## SFT Stage 1 With Cold Start Data

To generate proper cold start data to build R1 using Supervised fine-tuning, we obviously need a proper team along with an excessive amount of code, but thankfully, we already have data ([Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)) that is similar to the cold start form.
> We need to know what and how training happens inside the SFT Trainer as it processes our training data?

SFT is a form of supervised learning. This means we’re giving the model pairs of inputs and *desired* outputs.

In our case, the input might be a problem prompt, and the desired output is the well-reasoned, step-by-step solution from our training dataset. **I hope this point gives a clear view of why there is a need of cold data.**

It takes our tokenized training data and feeds it to the model in batches. For each batch, a important set of operations happens, Let’s visualize this internal process:

![SFT WorkFlow (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6838/1*EsEgATw1aSYPjfGtpId2mQ.png)

First, the model takes an input, a problem prompt, for instance. It processes this input and generates its best guess for the solution, token by token. These are the *predicted tokens*.

Next, the SFT Trainer needs to know how good (or bad) these predictions are. It uses a *loss function*, typically Cross-Entropy Loss. This function mathematically compares the model’s predicted tokens to the *correct* tokens from our training data. Think of it as calculating the “error” of the model’s answer.

This “error” isn’t just discarded. It’s the crucial signal for learning. Through a process called *backpropagation*, this error is used to calculate *gradients*. Gradients are like guides, pointing in the direction of parameter adjustments that would reduce the error.

Finally, an *optimizer*, like **AdamW** uses these gradients to subtly tweak the model’s internal settings — its parameters. These tweaks are designed to make the model’s next prediction a little bit closer to the correct answer.

## Stage 1 SFT Trainer Configs for R1

Remember those problems we had with R1 Zero messy reasoning and language mixing? SFT is designed to fix exactly that. By training on high-quality, refined data, we’re teaching the model:

* **Clear Reasoning Style**: To structure its “thinking” in a way that’s easy to read and follow.

* **Consistent Language**: To stick to one language within a response, avoiding confusing mixes.

We’re using the Bespoke-Stratos-17k dataset for SFT. As we saw earlier, it’s got 17,000 problems focused on math and code, with a format that looks pretty good for our needs.

Let’s quickly remind ourselves of a sample from Bespoke-Stratos-17k:
```python
# Load the "Bespoke-Stratos-17k" dataset from bespokelabs
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default")

# Access the first sample in the training set
bespoke_rl['train'][0]


#### OUTPUT ####
{
  'system': 'Your role as an assistant involves thoroughly exploring questions thro ...',
  'conversations': [{'from': 'user', 'value': 'Return your final response within \\boxed{}. The operation $\\otimes$ is ...'}, {'from': 'assistant', 'value': '<|begin_of_thought|>  Okay, let me try to figure out this problem. So, ...'}],
}
#### OUTPUT ####
```

This dataset, with its system prompts and user-assistant conversations, is perfect for showing our model how conversations with reasoning should look.

We’ll use the trl library again, which makes SFT training super easy.

First, we need to set up our configurations, similar to what we did for GRPO, but this time for SFT.
```python
# Model and Output Configuration (same as before, or adjust as needed)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-SFT-training" # New output directory for SFT model
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SFT configuration - similar to GRPO (SFTConfig also extends TrainingArguments), but adjusted for SFT
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,             # Adjust epochs as needed
    max_steps=200,                  # Stop after 200 steps (remove it to train the whole epoch)
    per_device_train_batch_size=2,  # The reasoning traces are long, so fewer per device...
    gradient_accumulation_steps=8,  # ...and more accumulation (16 conversations per step)
    learning_rate=2e-5,             # Adjust learning rate for SFT
    warmup_steps=0.1,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    dataloader_num_workers=2,
    seed=42,
    bf16=True,
    push_to_hub=False,
    gradient_checkpointing=True,
    report_to="none",
    max_length=4096,                # Truncate conversations longer than 4096 tokens
)
```

These settings are quite similar to what we used for GRPO, with a few tweaks that suit SFT: a different learning rate, fewer conversations per device with more gradient accumulation (the reasoning traces are long), and max_length=4096, which truncates the longest conversations. Packing several short conversations into one sequence would be faster, but TRL needs FlashAttention for that, so we leave it off.

## Stage 1 STF Training Loop

Now, let’s load our dataset and tokenizer. We use the [HuggingFaceH4 copy of Bespoke-Stratos-17k](https://huggingface.co/datasets/HuggingFaceH4/Bespoke-Stratos-17k), which has the same conversations in the `messages` format (a list of `role`/`content` dictionaries) that SFTTrainer understands. Its `messages` hold the user and assistant turns only (the long system prompt stays in a separate column that we don't use), so the chat template adds Qwen's default system message instead.
```python
# Load Bespoke-Stratos-17k dataset
dataset_sft = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split='train') # Only using train split for simplicity

# Initialize tokenizer - same as before
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

And finally, we initialize the SFTTrainer and start training!
```python
# Initialize base model for SFT - same as before
model_sft = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16
)

# Initialize the SFT Trainer
sft_trainer = SFTTrainer(
    model=model_sft,                     # Our initialized Qwen model
    train_dataset=dataset_sft,           # Bespoke-Stratos-17k dataset
    processing_class=tokenizer,          # Tokenizer
    args=training_args,                  # SFTConfig defined above
    callbacks=callbacks,                 # Our logging callback from the GRPO part
)

# Start the SFT Training Loop
sft_train_result = sft_trainer.train()
```

When you run this code, you’ll see the SFT training process start. Our callback prints the loss and the learning rate every 10 steps:
```
INFO:__main__:Step 10: Loss = 1.2096, Learning Rate = 9.00e-06
INFO:__main__:Step 20: Loss = 1.0689, Learning Rate = 1.90e-05
INFO:__main__:Step 30: Loss = 1.0269, Learning Rate = 1.90e-05
INFO:__main__:Step 40: Loss = 0.9674, Learning Rate = 1.79e-05
INFO:__main__:Step 50: Loss = 0.9293, Learning Rate = 1.68e-05
...
INFO:__main__:Step 100: Loss = 0.9224, Learning Rate = 1.12e-05
...
INFO:__main__:Step 150: Loss = 0.9425, Learning Rate = 5.67e-06
...
INFO:__main__:Step 200: Loss = 0.9157, Learning Rate = 1.11e-07
```
The loss falls from 1.21 at step 10 to 0.92 at step 200. The 200 steps took 19 minutes on one RTX A6000 and used at most 5.9 GB of GPU memory. They cover 3,200 of the 16,610 conversations; a whole epoch is about 1,040 steps. 43% of the conversations are longer than 4,096 tokens, so they lose their ending, where the final solution is, to truncation.

## Saving Tiny R1 LLM

After SFT is done, we save our newly fine-tuned model (R1).
```python
# Saving the Trained SFT Model
TRAINED_SFT_MODEL_PATH = "data/Qwen-SFT-training" # Same as OUTPUT_DIR

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_SFT_MODEL_PATH)

# Save the trained model
sft_trainer.save_model(TRAINED_SFT_MODEL_PATH)

print(f"SFT Trained model saved to {TRAINED_SFT_MODEL_PATH}")


#### OUTPUT ####
SFT Trained model saved to data/Qwen-SFT-training
#### OUTPUT ####
```

Let’s ask our SFT model the same test problem we gave the GRPO model:
```python
# Testing Inference with the SFT Model
sft_model = sft_trainer.model.eval()  # eval mode: no dropout, and generate() can use its cache

# The training conversations have no system message, so the chat template adds Qwen's default one,
# both during training and here
messages = [{"role": "user", "content": dataset['test'][0]['problem']}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(sft_model.device)
outputs = sft_model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)

print(f"Test Input: {dataset['test'][0]['problem']}")
print(f"SFT Model Response: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)}")


#### OUTPUT ####
Test Input: In 1988, a person's age was equal to the sum of the digits of their birth year. How old was this person?
SFT Model Response: <|begin_of_thought|>

Okay, let me try to figure out how old someone in 1988 was based on their age being equal to the sum of the digits of their birth year. Hmm, so first, I need to find out what their birth year was. Since they were born in 1988, that would mean it's my year of birth as well. So if they were 52 years old in 1988 (since 1988 - 1987 = 1), then the sum of the digits would be 5 + 2 = 7. Therefore, my age is 7.

But wait, I don't know when or where I was born. Maybe I'm thinking about something else here. Let me think again. The problem says "their age was equal to the sum of the digits of their birth year." So maybe there's another person with the same birth year and their age is 7, but I have no idea who that might be. 

Alternatively, maybe I should look for someone whose birth year is 1988 and whose age is also 7. But since I can't recall any such person, maybe I need to check other possibilities?
...

Both conditions match: the birth year is 1936 and the digit sum is 19. Hence, the person's age is 52, which confirms our solution.

Thus, the final answer is \(\boxed{1981}\).

<|end_of_solution|>
#### OUTPUT ####
```
After 200 steps the model writes like the Bespoke-Stratos traces: it opens with `<|begin_of_thought|>`, second-guesses itself (“But wait”, “Let me think again”) and ends with a `<|begin_of_solution|>` section and a boxed answer. The reasoning is muddled and the answer is wrong (the person was 22, not 1981): 200 steps on 3,200 conversations teach the format, not the math.

And that’s it for the SFT part!We’ve now taken our base model, shown it lots of examples of good reasoning, and fine-tuned it to be better at producing clear, structured responses.
> This finetuned model using SFT is what we called R1 after SFT stage 1

The steps after SFT, especially the RL stages and rejection sampling, are complex to implement from scratch in Python. Focusing on the theoretical understanding is key to understand the overall process.

## Reasoning-Oriented RL

After SFT, the model can reason better, but we want to *really* focus on reasoning quality and fix language mixing. This stage uses RL again, but with a smarter reward system.

This new reward checks if the model reasoning and answer are in the same language as the question. If you ask in English, the *whole* response should be in English. This fixes language mixing issues.

![Reasoning Oriented RL (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7468/1*Z2oHDdkWb7RnO5uVHPSvMg.png)

It adds a **Language Consistency Reward** alongside accuracy to ensure the SFT model reasons and answers in the same language as the input.

The GRPO algorithm and training loop from R1 Zero are reused, but the reward signals are improved to specifically target better reasoning and consistent language output.

## Rejection Sampling

To get super high-quality reasoning data, DeepSeek uses **Rejection Sampling**. Think of it as a filter to keep only the *best* examples.

![Rejection Sampling (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/8520/1*obG-BrhwtIuOv7YBZIpSwg.png)

The model generates many reasoning examples. These are then evaluated for correctness and reasoning quality (often using a generative reward model and human checks).

Only the *best*, high-quality reasoning examples are kept. Combined with non-reasoning data, this refined dataset is used for a second **SFT Stage 2**, further improving reasoning and general abilities.

## SFT Stage 2 Training

The final RL stage focuses on making the model a helpful and safe AI assistant for *all* situations, not just reasoning problems. This is about alignment with human values.

**Key Focus: Helpfulness & Harmlessness Rewards**

Not just accuracy, the reward system now includes:

* **Helpfulness:** Is the response useful and informative?

* **Harmlessness:** Is the response safe, unbiased, and ethical?

![SFT Stage 2 (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7086/1*_u5ALx4VYQpsSgT_0s10HQ.png)

The training data becomes diverse, including reasoning tasks and human preference data (which output is better — more helpful, less harmful?).

The reward system now balances accuracy with **helpfulness and harmlessness**. Iterative RL training (likely GRPO again) optimizes the model to be not just good at reasoning, but also a safe and helpful AI assistant for general use, resulting in DeepSeek R1.

## Distillation

To make DeepSeek R1 accessible, they **distilled** its knowledge into smaller models.

![Distillation Process (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/2500/0*QdOxtvuKaEASreK0.png)

Distillation takes the knowledge of a large, powerful “teacher” model (DeepSeek R1) and transfers it to smaller “student” models. Using a large dataset of reasoning examples, the outputs of DeepSeek R1 are used as the *target* answers.

Smaller models are then trained (SFT) to mimic these outputs. This results in smaller, faster models that retain a significant portion of DeepSeek R1’s reasoning abilities, making them more practical for wider use.

Happy reading!

## Citation

This project was completed with the help of the OpenR1 implementation from Hugging Face.

```
@misc{openr1,
    title = {Open R1: A fully open reproduction of DeepSeek-R1},
    url = {https://github.com/huggingface/open-r1},
    author = {Hugging Face},
    month = {January},
    year = {2025}
}
```
