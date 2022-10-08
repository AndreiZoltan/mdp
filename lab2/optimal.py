from argparse import ArgumentParser
from networkx import DiGraph
import numpy as np
from scipy.optimize import linprog

from prettytable import PrettyTable


def build_parser() -> ArgumentParser:
    argparser = ArgumentParser(prog="python optimal.py")
    argparser.add_argument(
        "--path", default="graph.csv", type=str, help="path to your csv file"
    )
    argparser.add_argument(
        "--eps", default="1e-6", type=float, help="epsilon to reset small values"
    )
    return argparser


def create_graph(graph_table: np.array, graph: DiGraph) -> None:
    for i in range(graph_table.shape[0]):
        for j in range(graph_table.shape[1]):
            if graph_table[i, j] != "nan":
                c, t = graph_table[i, j].split(",")
                graph.add_edge(i, j, c=float(c), t=float(t))


def get_optimal_path(cycle: DiGraph, x: np.array, z_edge: np.array) -> list:
    path = list()
    for edge in np.argwhere(x > 0):
        cycle.add_edge(*z_edge[edge][0])
    first = list(cycle.nodes)[0]
    path.append(first)
    state = first
    node = None
    while node != first:
        node = list(cycle.neighbors(state))[0]
        path.append(node)
        state = node
    return path


def main(args):
    graph_table = np.genfromtxt(
        args.path, delimiter=";", dtype="str", skip_header=False
    )
    graph = DiGraph()
    create_graph(graph_table[1:], graph)

    cycle = DiGraph()
    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
    z = np.empty((n_edges,))
    z_edges = np.empty((0, 2), int)
    a_ub = np.zeros((n_nodes, n_edges))
    b_ub = np.zeros((n_nodes, 1))
    a_eq = np.zeros((1, n_edges))
    b_eq = np.ones(1)
    i = 0
    for node in graph.nodes:
        for adj_n, datadict in graph.adj[node].items():
            z[i] = datadict["c"]
            z_edges = np.append(z_edges, np.array([[node, adj_n]]), axis=0)
            a_ub[node, i] = 1
            a_ub[adj_n, i] = -1
            a_eq[0, i] = datadict["t"]
            i += 1
    solution = linprog(z, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=(0, None))
    fval = solution["fun"]
    x = np.array(solution["x"])
    x[abs(x) < args.eps] = 0.0
    x_path = get_optimal_path(cycle, x, z_edges)
    table = PrettyTable()
    table.field_names = np.array(["cost", "from node", "to node"])
    table.add_rows(np.append(np.expand_dims(np.round(x, 3), axis=1), z_edges, axis=1))
    print(table)
    print("Optimal average cost for infinity is {}".format(np.round(fval, 3)))
    print("Optimal path for an agent is: ", end="")
    print(*x_path, sep=" -> ")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
