from argparse import ArgumentParser
from networkx import DiGraph
import numpy as np

from prettytable import PrettyTable


def build_parser() -> ArgumentParser:
    argparser = ArgumentParser(prog="python optimal.py")
    argparser.add_argument(
        "--tmin", type=int, default=5, help="the minimal time from which the cost is calculated"
    )
    argparser.add_argument(
        "--tmax", type=int, default=10, help="the maximum time until which the cost is calculated"
    )
    argparser.add_argument("--x0", default=0, type=int, help="node to start")
    argparser.add_argument("--xf", default=7, type=int, help="node to finish")
    argparser.add_argument(
        "--print",
        action="store_true",
        required=False,
        help="print pretty table for you",
    )
    argparser.add_argument(
        "--path", default="graph.csv", type=str, help="path to your csv file"
    )
    return argparser


class TransitionCost:
    def __init__(self, func_expr: str):
        self.ast = compile(func_expr, "<string>", "eval")

    def __call__(self, t):
        return eval(self.ast)


def create_graph(graph_table: np.array, graph: DiGraph) -> None:
    for i in range(graph_table.shape[0]):
        for j in range(graph_table.shape[1]):
            if graph_table[i, j] != "nan":
                graph.add_edge(i, j, c=TransitionCost(graph_table[i, j]))


def get_optimal_path(graph: DiGraph, table: np.array, x_f: int, f_min: int) -> list:
    path = [x_f]
    last = x_f
    t_f = np.where(table[:, x_f] == f_min)[0][0]
    for t in range(t_f, 0, -1):
        for predecessor in graph.predecessors(last):
            if table[t][last] == table[t - 1][predecessor] + graph[predecessor][last]["c"](t - 1):
                path.append(predecessor)
                last = predecessor
                break

    return path


def main(args):
    graph_table = np.genfromtxt(
        args.path, delimiter=",", dtype="str", skip_header=False
    )
    graph = DiGraph()
    create_graph(graph_table[1:], graph)

    n_nodes = graph.number_of_nodes()
    opt_table = np.full((1, n_nodes), np.inf)
    opt_table[0, args.x0] = 0
    states = {args.x0}
    next_states = set()
    for t in range(1, args.tmax + 1):
        opt_table = np.append(opt_table, np.full((1, n_nodes), np.inf), axis=0)
        for state in states:
            next_states.update(graph.successors(state))
            for successor in graph.successors(state):
                opt_table[-1, successor] = min(
                    opt_table[-2, state] + graph[state][successor]["c"](t - 1),
                    opt_table[-1, successor],
                )
        states = next_states
        next_states = set()

    f_min = (
        np.min(opt_table[args.tmin, args.xf])
        if args.tmin == args.tmax
        else np.min(opt_table[args.tmin: args.tmax, args.xf])
    )
    x_path = get_optimal_path(graph, opt_table, args.xf, f_min)

    if args.print:
        table = PrettyTable()
        table.field_names = np.append("t\\x", graph_table[0])
        table.add_rows(
            np.append(
                np.expand_dims(np.arange(opt_table.shape[0]), axis=1), opt_table, axis=1
            )
        )
        print(table)

    print("Optimal cost for given T_min and T_max is {}".format(f_min))
    print("Optimal path for an agent is: ", end="")
    print(*x_path[::-1], sep=" -> ")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
