# Monte-Carlo Tree Search for Large Language Models

This repository is a fork of [Dyna Gym](https://github.com/SuReLI/dyna-gym) and extends its functionality to focus on using Monte-Carlo tree search for decoding large language models (LLMs).

## Installation

First, create a new Conda environment (optional):

```bash
conda create --name mcts-for-llm python=3.10
conda activate mcts-for-llm
```
We tested on python 3.10.0. Other versions may work as well.

Then, git clone this repo and install the package:

```bash
pip install -e .
```

## Examples

### Using GPT-2 and UCT for Language Alignment with Positive Sentiment Reward

Run the following command to generate texts using the GPT-2 model, guided by UCT (Upper Confidence Bound applied to Trees) for language alignment. Positive sentiment is used as the reward.

```bash
python examples/uct_language_alignment.py
```

### Classic Planning Domains (Non-LLM)

This repository also includes some classic planning domains derived from the original Dyna Gym repo. These examples don't use LLMs but may be useful for debugging purposes.

```bash
python examples/uct_nscartpole_v0.py
python examples/uct_nscartpole_v1.py
...
```
