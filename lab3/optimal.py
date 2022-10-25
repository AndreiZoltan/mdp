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
                data = graph_table[i, j]
                if "," in data:
                    p, c = data.split(",")
                    graph.add_edge(i, j, p=float(p), c=float(c))
                    graph.nodes[i]["control"] = False
                else:
                    graph.add_edge(i, j, c=float(data))
                    graph.nodes[i]["control"] = True


def get_n_control_edges(graph: DiGraph) -> tuple[int, int]:
    n = 0
    c_nodes = []
    for node in graph.nodes(data=True):
        if node[1]["control"]:
            c_nodes.append(node[0])
    return len(c_nodes), len(graph.edges(c_nodes))


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
    print(graph.in_edges(0))
    n_control, n_control_edges = get_n_control_edges(graph)
    n_not_control = graph.number_of_nodes() - n_control

    z = np.zeros((n_control_edges + n_not_control))

    a_eq = np.zeros((graph.number_of_nodes(), n_control_edges + n_not_control))
    a_eq = np.append(a_eq, np.ones((1, n_control_edges + n_not_control)), axis=0)

    b_eq = np.zeros(graph.number_of_nodes() + 1)
    b_eq[-1] = 1

    cycle = DiGraph()
    z_edges = np.empty((0, 2), int)
    j = 0
    control_dict = {
        node[0]: i
        for i, node in enumerate(
            sorted(graph.nodes(data=True), key=lambda x: (~x[1]["control"], x[0]))[:n_control]
        )
    }
    not_control_dict = {
        node[0]: i
        for i, node in enumerate(
            sorted(graph.nodes(data=True), key=lambda x: (x[1]["control"], x[0]))[:n_not_control]
        )
    }
    for i, node in enumerate(
        sorted(graph.nodes(data=True), key=lambda x: (~x[1]["control"], x[0]))[:n_control]
    ):
        for edge in sorted(graph.edges(node[0], data=True)):
            z[j] = edge[2]["c"]
            print(graph.nodes[edge[1]])
            a_eq[i, j] = -1
            if graph.nodes[edge[1]]["control"]:
                a_eq[control_dict[edge[1]], j] = 1
            else:
                a_eq[n_control+not_control_dict[edge[1]], j] = 1
            for in_edge in graph.in_edges(edge[0], data=True):
                if not graph.nodes[in_edge[0]]["control"]:
                    print(in_edge)
                    a_eq[i, n_control_edges+not_control_dict[in_edge[0]]] = in_edge[2]["p"]
            j += 1
    for i, node in enumerate(
        sorted(graph.nodes(data=True), key=lambda x: (~x[1]["control"], x[0]))[n_control:]
    ):
        mu = 0
        a_eq[n_control+not_control_dict[node[0]], n_control_edges+not_control_dict[node[0]]] = -1
        for edge in sorted(graph.edges(node[0], data=True)):
            mu += edge[2]["p"]*edge[2]["c"]
            if not graph.nodes[edge[1]]["control"]:
                a_eq[n_control+not_control_dict[edge[1]], n_control_edges+not_control_dict[edge[0]]] = edge[2]["p"]
        z[j] = mu
        j += 1
    # print(z, "\n", a_eq, "\n", b_eq)

    solution = linprog(z, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    print(solution)
    # fval = solution["fun"]
    # x = np.array(solution["x"])
    # x[abs(x) < args.eps] = 0.0
    # x_path = get_optimal_path(cycle, x, z_edges)
    # table = PrettyTable()
    # table.field_names = np.array(["cost", "from node", "to node"])
    # table.add_rows(np.append(np.expand_dims(np.round(x, 3), axis=1), z_edges, axis=1))
    # print(table)
    # print("Optimal average cost for infinity is {}".format(np.round(fval, 3)))
    # print("Optimal path for an agent is: ", end="")
    # print(*x_path, sep=" -> ")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
