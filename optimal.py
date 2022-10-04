import argparse
from networkx import DiGraph
import numpy as np

from prettytable import PrettyTable

from typing import Callable


def build_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(prog="python optimal.py")
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


def func(func_expr: str) -> Callable[[int], float]:
    """
    parse string expression and return c function
    :param func_expr: expression of transition function
    :return: c function
    """

    class TransitionCost:
        def __init__(self, ast_):
            self.ast = ast_

        def __call__(self, t: int):
            return eval(ast)

    ast = compile(func_expr, "<string>", "eval")
    return TransitionCost(ast)


def create_graph(graph_table: np.array, graph: DiGraph) -> None:
    """
    builds graph from table data
    :param graph_table:
    :param graph:
    :return:
    """
    for i in range(graph_table.shape[0]):
        for j in range(graph_table.shape[1]):
            if graph_table[i, j] != "nan":
                graph.add_edge(i, j, c=func(graph_table[i, j]))


def get_optimal_path(graph: DiGraph, table: np.array, x_f: int) -> list:
    path = [x_f]
    last = x_f
    for t in range(table.shape[0] - 1, 0, -1):
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
            next_states.update(set(graph.successors(state)))
            for successor in graph.successors(state):
                opt_table[-1, successor] = min(
                    opt_table[-2, state] + graph[state][successor]["c"](t - 1),
                    opt_table[-1, successor],
                )
        states = next_states
        next_states = set()

    x_path = get_optimal_path(graph, opt_table, args.xf)
    f_min = (
        np.min(opt_table[args.tmin, args.xf])
        if args.tmin == args.tmax
        else np.min(opt_table[args.tmin: args.tmax, args.xf])
    )

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
