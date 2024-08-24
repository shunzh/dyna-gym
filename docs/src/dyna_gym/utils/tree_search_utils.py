import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from dyna_gym.agents.mcts import DecisionNode, ChanceNode, chance_node_value


def update_root(ag, act, state_p):
    root_updated = False
    for chance_node in ag.root.children:
        if act == chance_node.action:
            for decision_node in chance_node.children:
                if decision_node.state == state_p:
                    ag.root = decision_node
                    root_updated = True
                    break

    if not root_updated:
        raise Exception("root update fails, can't find the next state, action pair in tree.")


def pre_order_traverse(
        decision_node: DecisionNode,
        decision_node_fn=lambda n, d: None,
        chance_node_fn=lambda n, d: None,
        depth=0):
    """
    Postorder traversal of the tree rooted at state
    Apply fn once visited
    """
    decision_node_fn(decision_node, depth)

    for chance_node in decision_node.children:
        chance_node_fn(chance_node, depth)
        for next_decision_node in chance_node.children:
            pre_order_traverse(next_decision_node, decision_node_fn, chance_node_fn, depth + 1)


def get_all_decision_nodes(root: DecisionNode):
    """
    Get all decision nodes in the tree
    """
    decision_nodes = []
    pre_order_traverse(root, decision_node_fn=lambda n, d: decision_nodes.append(n))
    return decision_nodes


def print_tree(root: DecisionNode, tokenizer):
    def printer(node: ChanceNode, depth):
        # print the average return of the *parent* of this state
        # (this is easier to implement than printing all its children nodes)
        print("\t" * depth,
              repr(tokenizer.decode(node.action)),
              'prob', node.prob,
              'returns', node.sampled_returns)

    pre_order_traverse(root, chance_node_fn=printer)


def plot_tree(root: DecisionNode, tokenizer, filename):
    """
    Plot the tree rooted at root
    """
    # plot the tree
    G = nx.DiGraph()
    G.add_node(root.id, label='<PD>')

    def add_node(node: ChanceNode, depth):
        if len(node.children) > 0:
            child_id = node.children[0].id
            parent_id = node.parent.id

            G.add_node(child_id)

            avg_return = np.mean(node.sampled_returns)
            edge_label = f'{repr(tokenizer.decode(node.action))}\np={node.prob:.2f}\nR={avg_return:.2f}'
            G.add_edge(parent_id, child_id, label=edge_label)

    pre_order_traverse(root, chance_node_fn=add_node)

    plt.figure(figsize=(15, 15))

    pos = hierarchy_pos(G, root=root.id)
    nx.draw(G, pos, with_labels=True)

    edge_labels = nx.get_edge_attributes(G, 'label')
    # plot labels on the edges horizontally
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, rotate=False)

    plt.savefig(filename + '.pdf', format="pdf")
    plt.close()


def convert_to_json(root: DecisionNode, env, selected_act):
    """
    A function to serialize a tree and return a json object
    """
    ret = []

    def get_info(node: ChanceNode, depth):
        if node.action == env.terminal_token:
            # terminal state has no complete_program attribute, since the program is already complete
            complete_program = env.convert_state_to_program(node.children[0].state)
        else:
            complete_program = env.convert_state_to_program(node.children[0].info['complete_program'])

        info = {'token': env.tokenizer.decode(node.action),
                'state': env.convert_state_to_program(node.children[0].state),
                'selected': node.action == selected_act,
                'score': chance_node_value(node),
                'complete_program': complete_program}
        ret.append(info)

    pre_order_traverse(root, chance_node_fn=get_info)
    return ret


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, leaf_vs_root_factor=0.5):
    """
    Shun: As of early 2023, I couldn't find a layout in graphviz that plots a tree nicely.
    So I'm using the following function found in this answer:
    https://stackoverflow.com/a/29597209/1025757

    ---
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.


    :Arguments:

    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx=0.2, vert_gap=0.2, vert_loc=0,
                       xcenter=0.5, rootpos=None,
                       leafpos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root: (xcenter, vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            rootdx = width / len(children)
            nextx = xcenter - width / 2 - rootdx / 2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G, child, leftmost + leaf_count * leafdx,
                                                             width=rootdx, leafdx=leafdx,
                                                             vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                                             xcenter=nextx, rootpos=rootpos, leafpos=leafpos,
                                                             parent=root)
                leaf_count += newleaves

            leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
        #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
        #        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width / 2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node) == 0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node) == 1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width,
                                                  leafdx=width * 1. / leafcount,
                                                  vert_gap=vert_gap,
                                                  vert_loc=vert_loc,
                                                  xcenter=xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (
        leaf_vs_root_factor * leafpos[node][0] + (1 - leaf_vs_root_factor) * rootpos[node][0], leafpos[node][1])
    #    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / xmax, pos[node][1])
    return pos
