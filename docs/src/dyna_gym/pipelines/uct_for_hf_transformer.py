from datetime import datetime
from typing import Callable, Sequence

import gym
import torch
import transformers

from dyna_gym.agents import uct
from dyna_gym.default_policy.hf_default_policy import HuggingFaceDefaultPolicy
from dyna_gym.utils.tree_search_utils import print_tree


def uct_for_hf_transformer_pipeline(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        horizon: int = 100,
        reward_func: Callable = None,
        uct_args: dict = {},
        model_generation_args: dict = {},
        should_plot_tree: bool = False,
        reward_func_input_is_state: bool = False,
) -> Callable:
    """
    A wrapped UCT agent for HuggingFace transformer.

    Args:
        model_name: The name of a HuggingFace transformer model. If provided, will load the model and tokenizer.
        model: A HuggingFace transformer model.
        tokenizer: A HuggingFace tokenizer.
        horizon: The maximum number of steps to take.
        reward_func: A function that evaluate the reward of a sequence.
        value_func: A function that evaluate the value of a sequence.
        uct_args: Arguments for the UCT agent.
        model_generation_args: Arguments for the model generation.
        should_plot_tree: Whether to plot the tree after generation.
        reward_func_input_is_state: Whether the input of the reward function is (token ids, attention masks) or tokenized text.
    """
    eos_token_id = tokenizer.eos_token_id

    if not reward_func_input_is_state:
        # reward function takes tokenized text as input, decode tokeni ids here
        # if reward function takes token ids as input, this step can be saved
        def reward_func_(state):
            ids, attention_mask = state
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return reward_func(text)
    else:
        reward_func_ = reward_func

    env = gym.make(
        'LanguageEnv-v0',
        terminal_token=eos_token_id,
        horizon=horizon,
        reward_func=reward_func_,
    )

    default_policy = HuggingFaceDefaultPolicy(
        env=env,
        horizon=horizon,
        model=model,
        generation_args=model_generation_args,
    )

    agent = uct.UCT(
        default_policy=default_policy,
        **uct_args
    )

    ### Run
    def generate(input_ids=None, input_str=None, attention_mask=None):
        assert (input_ids is None) != (input_str is None), "Only provide one of input_ids and input_str"

        if input_str is not None:
            input_ids = tokenizer.encode(input_str)
            input_ids = torch.tensor(input_ids).to(model.device)

        if attention_mask is None:
            # attend to tokens that are not padding
            if tokenizer.pad_token_id is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            attention_mask = attention_mask.to(model.device)

        env.reset(input_ids, attention_mask)
        # do all rollouts in one step
        env.step(agent.act(env, done=False))

        # print tree
        print_tree(agent.root, tokenizer)
        # optionally, plot the tree and save to a pdf file
        if should_plot_tree:
            # plot (and print) the tree
            from dyna_gym.utils.tree_search_utils import plot_tree
            filename = f"tree-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            plot_tree(agent.root, tokenizer, filename)
            print(f"Tree plotted and saved to {filename}.pdf")

        results = {
            'output_ids': agent.rolled_out_trajectories,
            'rewards': agent.rolled_out_rewards,
            'texts': [tokenizer.decode(ids, skip_special_tokens=True) for ids in agent.rolled_out_trajectories],
        }

        # clear for the next generation call
        agent.reset()

        return results

    return generate
