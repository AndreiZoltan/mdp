from argparse import ArgumentParser
import numpy as np
from networkx import DiGraph

from mdp import MDP


def build_parser() -> ArgumentParser:
    argparser = ArgumentParser(prog="python optimal.py")
    argparser.add_argument("--x0", default=0, type=int, help="node to start")
    argparser.add_argument("--gamma", default="0.5", type=float, help="gamma discount")
    argparser.add_argument(
        "--path", default="graph.csv", type=str, help="path to your csv file"
    )
    argparser.add_argument(
        "--verbose", default=False, action="store_true", help="verbose output"
    )
    return argparser


def create_graph(graph_table: np.array, graph: DiGraph) -> None:
    for i in range(graph_table.shape[0]):
        for j in range(graph_table.shape[1]):
            if graph_table[i, j] != "nan":
                data = graph_table[i, j]
                if "," in data:
                    p, c = data.split(",")
                    graph.add_edge(i, j, p=float(p), c=float(c))
                    graph.nodes[i]["control"] = False
                else:
                    graph.add_edge(i, j, c=float(data))
                    graph.nodes[i]["control"] = True


def parse_graph(path):
    graph_table = np.genfromtxt(path, delimiter=";", dtype="str", skip_header=False)
    graph = DiGraph()
    create_graph(graph_table[1:], graph)

    def all_states(_graph):
        return set(node[0] for node in _graph.nodes(data=True) if node[1]["control"])

    def actions(state, _graph):
        return set(
            node
            for node in _graph.successors(state)
            if not _graph.nodes[node]["control"]
        )

    def states(action, _graph):
        return {
            node for node in _graph.successors(action) if _graph.nodes[node]["control"]
        }

    graph_dict = {
        state: {
            action: {
                state: graph.edges[action, state]["p"]
                for state in states(action, graph)
            }
            for action in actions(state, graph)
        }
        for state in all_states(graph)
    }
    rewards = {
        state: {
            action: {
                state: graph.edges[action, state]["c"]
                for state in states(action, graph)
            }
            for action in actions(state, graph)
        }
        for state in all_states(graph)
    }
    return graph_dict, rewards


def get_action_value(mdp, state_values, state, action, gamma):
    action_values = [
        mdp.get_transition_prob(state, action, next_state)
        * (mdp.get_reward(state, action, next_state) + gamma * state_values[next_state])
        for next_state in mdp.get_all_states()
    ]
    return np.sum(action_values)


def get_new_state_value(mdp, state_values, state, gamma):
    if mdp.is_terminal(state):
        return 0
    q_values = [
        get_action_value(mdp, state_values, state, action, gamma)
        for action in mdp.get_possible_actions(state)
    ]
    return np.max(q_values)


def get_state_values(mdp, gamma, num_iter, min_difference, verbose=False):
    state_values = {s: 0 for s in mdp.get_all_states()}
    for i in range(num_iter):
        new_state_values = {
            state: get_new_state_value(mdp, state_values, state, gamma)
            for state in state_values.keys()
        }
        diff = max(
            abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states()
        )
        if verbose:
            print("iter %4i   |   diff: %6.5f   |   " % (i, diff), end="")
            print("   ".join("V(%s) = %.3f" % (s, v) for s, v in state_values.items()))
        state_values = new_state_values
        if diff < min_difference:
            if verbose:
                print("Terminated")
            break
    return state_values


def get_optimal_action(mdp, state_values, state, gamma=0.9):
    if mdp.is_terminal(state):
        return None
    q_values = {
        action: get_action_value(mdp, state_values, state, action, gamma)
        for action in mdp.get_possible_actions(state)
    }

    return sorted(q_values.items(), key=lambda x: x[1])[-1][0]


def main(args):
    graph_dict, rewards = parse_graph(args.path)
    mdp = MDP(graph_dict, rewards, initial_state=0)
    gamma = 0.9
    num_iter = 100
    min_difference = 0.001
    state_values = get_state_values(mdp, gamma, num_iter, min_difference, args.verbose)
    s = mdp.reset()
    rewards = []
    for _ in range(10000):
        s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)
    print("average reward: ", np.mean(rewards))

    for state in state_values.keys():
        print(
            "For {} state {} is optimal".format(
                state, (state, get_optimal_action(mdp, state_values, state, gamma))
            )
        )


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
