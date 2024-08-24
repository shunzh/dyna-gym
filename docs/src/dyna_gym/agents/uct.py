"""
UCT Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s, a, is_model_dynamic)
env.equality_operator(s1, s2)
"""

import dyna_gym.agents.mcts as mcts
from dyna_gym.utils.utils import combinations
from math import sqrt, log
from gym import spaces


class UCT(object):
    """
    UCT agent
    """
    def __init__(
            self,
            action_space=[],
            rollouts=100,
            horizon=100,
            gamma=0.9,
            ucb_constant=6.36396103068,
            ucb_base=50.,
            is_model_dynamic=True,
            width=None,
            default_policy=None,
            ts_mode='sample',
            reuse_tree=False,
            alg='uct',
            lambda_coeff=0.,
            value_func=None,
    ):
        """
        Args:
            action_space: action space of the environment
            rollouts: total number of rollouts (the number of loops over selection, expansion, simulation and backpropagation)
            horizon: maximum length of the rollouts
            gamma: discount factor
            ucb_constant: constant for the UCB exploration
            ucb_base: base for the UCB exploration, only used in var_p_uct
            is_model_dynamic: whether the model is dynamic
            width: number of children for each node, default is num of actions
            default_policy: an optional default policy that returns a most-likely sequence and top-k most-likely next tokens
            ts_mode: the mode for tree search, can be 'sample', 'best'
            reuse_tree: whether to reuse the tree from the previous step if the algorithm is called multiple times
            alg: exact UCT algorithm to use, can be 'uct', 'p_uct', 'var_p_uct'
        """
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic
        # the number of children for each node, default is num of actions
        self.width = width or self.n_actions
        self.default_policy = default_policy
        self.ts_mode = ts_mode
        self.reuse_tree = reuse_tree

        act_selection_criteria = {
            'uct': self.ucb,
            'p_uct': self.p_ucb,
            'var_p_uct': self.var_p_ucb
        }
        if alg in ['uct', 'p_uct', 'var_p_uct']:
            self.tree_policy = lambda children: max(children, key=act_selection_criteria[alg])

            if alg == 'var_p_uct':
                self.ucb_base = ucb_base
        else:
            raise Exception(f'unknown uct alg {alg}')

        self.lambda_coeff = lambda_coeff
        self.value_func = value_func

        self.reset()

    def reset(self):
        self.root = None
        self.rolled_out_trajectories = []
        self.rolled_out_rewards = []

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying UCT agent:')
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('UCB constant       :', self.ucb_constant)
        print('Is model dynamic   :', self.is_model_dynamic)
        print('Expansion Width    :', self.width)
        print()

    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
        return mcts.chance_node_value(node)\
            + self.ucb_constant * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, weighted by prior probability
        """
        return mcts.chance_node_value(node)\
            + self.ucb_constant * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def var_p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, the ucb exploration weight is a variable
        """
        ucb_parameter = log((node.parent.visits + self.ucb_base + 1) / self.ucb_base) + self.ucb_constant
        return mcts.chance_node_value(node)\
            + ucb_parameter * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def act(self, env, done, term_cond=None):
        root = self.root if self.reuse_tree else None
        opt_act, self.root = mcts.mcts_procedure(self, self.tree_policy, env, done, root=root, term_cond=term_cond)
        return opt_act
