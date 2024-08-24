from collections import OrderedDict

import gym
import torch


class LanguageEnv(gym.Env):
    """
    Langauge generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Transition: the next state is the current state concatenated with the action.
    Reward: an external function that evaluates a state (pass rate for programs, alignment score for natural language, etc.)
    Terminal state: the program reaches the maximum length or the terminal token is generated.
    """
    def __init__(self, terminal_token, horizon, reward_func):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon

        self.get_reward = reward_func

    def reset(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask
        else:
            attention_mask = torch.ones_like(input_ids)

        self.state = (input_ids, attention_mask)
        self.input_len = len(input_ids)
        return self.state

    def transition(self, s, a, is_model_dynamic=False):
        ids, attention_mask = s

        # s is a one-dimensional tensor, a is a token id (scalar), concatenate them to form a new state
        next_ids = torch.cat([ids, torch.tensor([a]).to(ids.device)])
        # append a 1 to the attention mask
        attention_mask = torch.cat([attention_mask, torch.tensor([1]).to(attention_mask.device)])

        if a == self.terminal_token or len(next_ids) == self.horizon:
            # either the text finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward((next_ids, attention_mask))
        else:
            reward = 0  # no intermediate reward

        return (next_ids, attention_mask), reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}

    def equality_operator(self, s1, s2):
        # s1 and s2 are two tensors
        return all(torch.equal(x1, x2) for x1, x2 in zip(s1, s2))
