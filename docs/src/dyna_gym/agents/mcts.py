"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""
import random
from gym import spaces
from tqdm import tqdm

from dyna_gym.utils.utils import combinations, multigpu_breakpoint


def chance_node_value(node, mode="best"):
    """
    Value of a chance node
    """
    if len(node.sampled_returns) == 0:
        return 0
    elif mode == "best":
        # max return (reasonable because the model is deterministic?)
        return max(node.sampled_returns)
    elif mode == "sample":
        # Use average return
        return sum(node.sampled_returns) / len(node.sampled_returns)
    else:
        raise Exception(f"Unknown tree search mode {mode}")


def mcts_tree_policy(children):
    return random.choice(children)


def mcts_procedure(ag, tree_policy, env, done, root=None, term_cond=None, ts_mode="sample"):
    """
    Compute the entire MCTS procedure wrt to the selected tree policy.
    Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
    and returning the one chosen by the tree policy.

    Args:
        ag: the agent
        tree_policy: the action selection policy
        env: the gym environment
        done: whether the current state is terminal
        root: the root of the tree, reuse the tree if not None, otherwise create a new tree
        term_cond: termination condition, if not None, the procedure will terminate when term_cond() is True
        ts_mode: the mode for tree search, can be 'sample', 'best'
    """
    decision_node_num = 0
    if root is not None:
        # if using existing tree, making sure the root is updated correctly
        assert root.state == env.state
    else:
        # create an empty tree
        root = DecisionNode(None, env.state, ag.action_space.copy(), done, default_policy=ag.default_policy, id=decision_node_num)
        decision_node_num += 1

    for _ in tqdm(range(ag.rollouts), desc="Rolling out", leave=False):
        if term_cond is not None and term_cond():
            break
        rewards = [] # Rewards collected along the tree for the current rollout
        node = root # Current node
        terminal = done

        # Selection
        select = True
        while select:
            if type(node) == DecisionNode: # DecisionNode
                if node.is_terminal:
                    select = False # Selected a terminal DecisionNode
                else:
                    node = tree_policy(node.children) # Move down the tree, node is now a ChanceNode
            else: # ChanceNode
                # Given s, a, sample s' ~ p(s'|s,a), also get the reward r(s,a,s') and whether s' is terminal
                state_p, reward, terminal = env.transition(node.parent.state, node.action, ag.is_model_dynamic)
                rewards.append(reward)

                new_state = True
                # find if s' is already in the tree, if so point node to the corresponding DecisionNode (and new_state=False)
                # if not, create a new DecisionNode for s' and point node to it
                for i in range(len(node.children)):
                    if env.equality_operator(node.children[i].state, state_p):
                        # s' is already in the tree
                        node = node.children[i]
                        new_state = False
                        break

                if new_state:
                    # Selected a node for expansion
                    select = False

                    # Expansion to create a new DecisionNode
                    new_node = DecisionNode(node, state_p, ag.action_space.copy(), terminal, default_policy=ag.default_policy, id=decision_node_num)
                    node.children.append(new_node)
                    decision_node_num += 1
                    node = node.children[-1]

        # Evaluation
        # now `rewards` collected all rewards in the ChanceNodes above this node
        assert(type(node) == DecisionNode)
        state = node.state
        current_state = state
        if ag.default_policy is None:
            t = 0
            estimate = 0
            while (not terminal) and (t < ag.horizon):
                action = env.action_space.sample()
                state, reward, terminal = env.transition(state, action, ag.is_model_dynamic)
                estimate += reward * (ag.gamma**t)
                t += 1
        else:
            if not node.is_terminal:
                # follow the default policy to get a terminal state
                state = ag.default_policy.get_predicted_sequence(state)
                estimate = env.get_reward(state)

                ag.rolled_out_trajectories.append(state[0])
                ag.rolled_out_rewards.append(estimate)
                # also save this to current nodes for possible visualization
                node.info['complete_program'] = state[0]
            else:
                # the rewards are defined on terminating actions, the terminal states have no rewards
                estimate = 0

        if ag.lambda_coeff > 0:
            assert ag.value_func is not None, "value_func must be provided if lambda_coeff > 0"

            state_ids = current_state[0]
            value = ag.value_func(state_ids)
            estimate = ag.lambda_coeff * value + (1 - ag.lambda_coeff) * estimate

        # Backpropagation
        node.visits += 1
        node = node.parent
        assert(type(node) == ChanceNode)
        while node:
            if len(rewards) != 0:
                estimate = rewards.pop() + ag.gamma * estimate
            node.sampled_returns.append(estimate)
            node.parent.visits += 1
            node = node.parent.parent

        # should finish backpropagating all the rewards
        assert len(rewards) == 0

    return max(root.children, key=lambda n: chance_node_value(n, mode=ts_mode)).action, root


class DecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        default_policy: default policy, used to prioritize and filter possible actions
    """
    def __init__(self, parent, state, possible_actions=[], is_terminal=False, default_policy=None, id=None):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        if default_policy is None:
            self.possible_actions = possible_actions
            random.shuffle(self.possible_actions)

            # if no default policy is provided, assume selection probability is uniform
            self.action_scores = [1.0 / len(self.possible_actions)] * len(self.possible_actions)
        else:
            # get possible actions from default policy
            top_k_predict, top_k_scores = default_policy.get_top_k_tokens(self.state)

            self.possible_actions = top_k_predict
            self.action_scores = top_k_scores

        # populate its children
        self.children = [ChanceNode(self, (act, score))
                         for act, score in zip(self.possible_actions, self.action_scores)]

        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}

    def is_fully_expanded(self):
        return all([child.expanded() for child in self.children])


class ChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """
    def __init__(self, parent, action_and_score):
        self.parent = parent
        self.action = action_and_score[0]
        self.depth = parent.depth
        self.children = []
        self.prob = action_and_score[1] # the probability that this action should be token, provided by default policy
        self.sampled_returns = []

    def expanded(self):
        return len(self.children) > 0


class MCTS(object):
    """
    MCTS agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, is_model_dynamic=True, default_policy=None):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.is_model_dynamic = is_model_dynamic
        self.default_policy = default_policy
        self.lambda_coeff = 0.0

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying MCTS agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('Is model dynamic   :', self.is_model_dynamic)

    def act(self, env, done):
        opt_act, _, = mcts_procedure(self, mcts_tree_policy, env, done)
        return opt_act
