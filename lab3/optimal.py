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
        "--print", help="print cost table", required=False, action="store_true"
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
    c_nodes = []
    for node in graph.nodes(data=True):
        if node[1]["control"]:
            c_nodes.append(node[0])
    return len(c_nodes), len(graph.edges(c_nodes))


def main(args):
    graph_table = np.genfromtxt(
        args.path, delimiter=";", dtype="str", skip_header=False
    )
    graph = DiGraph()
    create_graph(graph_table[1:], graph)
    n_control, n_control_edges = get_n_control_edges(graph)
    n_not_control = graph.number_of_nodes() - n_control

    z = np.zeros((n_control_edges + n_not_control))
    z_edges = np.zeros((n_control_edges, 2))
    a_eq = np.zeros((graph.number_of_nodes() + 1, n_control_edges + n_not_control))
    a_eq[-1] = np.ones((1, n_control_edges + n_not_control))
    b_eq = np.zeros(graph.number_of_nodes() + 1)
    b_eq[-1] = 1

    control_dict = {
        node[0]: i
        for i, node in enumerate(
            sorted(graph.nodes(data=True), key=lambda x: (~x[1]["control"], x[0]))[
                :n_control
            ]
        )
    }
    not_control_dict = {
        node[0]: i
        for i, node in enumerate(
            sorted(graph.nodes(data=True), key=lambda x: (x[1]["control"], x[0]))[
                :n_not_control
            ]
        )
    }

    j = 0
    for i, node in enumerate(
        sorted(graph.nodes(data=True), key=lambda x: (~x[1]["control"], x[0]))[:n_control]
    ):
        for edge in sorted(graph.edges(node[0], data=True)):
            z[j] = edge[2]["c"]
            z_edges[j] = edge[:2]
            a_eq[i, j] = -1
            if graph.nodes[edge[1]]["control"]:
                a_eq[control_dict[edge[1]], j] = 1
            else:
                a_eq[n_control + not_control_dict[edge[1]], j] = 1
            for in_edge in graph.in_edges(edge[0], data=True):
                if not graph.nodes[in_edge[0]]["control"]:
                    a_eq[i, n_control_edges + not_control_dict[in_edge[0]]] = in_edge[2]["p"]
            j += 1
    for i, node in enumerate(
        sorted(graph.nodes(data=True), key=lambda x: (~x[1]["control"], x[0]))[n_control:]
    ):
        mu = 0
        a_eq[
            n_control + not_control_dict[node[0]],
            n_control_edges + not_control_dict[node[0]],
        ] = -1
        for edge in sorted(graph.edges(node[0], data=True)):
            mu += edge[2]["p"] * edge[2]["c"]
            if not graph.nodes[edge[1]]["control"]:
                a_eq[
                    n_control + not_control_dict[edge[1]],
                    n_control_edges + not_control_dict[edge[0]],
                ] = edge[2]["p"]
        z[j] = mu
        j += 1

    result = linprog(z, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    fval = result["fun"]
    solution = np.array(result["x"])
    optimal_edges = z_edges[np.nonzero(solution[:n_control_edges])]

    table = PrettyTable()
    table.field_names = np.array(["cost", "from node", "to node"])
    table.add_rows(
        np.append(
            np.expand_dims(np.round(solution[:n_control_edges], 3), axis=1),
            z_edges,
            axis=1,
        )
    )
    if args.print:
        print(table)
    print("Optimal average cost for infinity is {}".format(np.round(fval, 3)))
    print("Optimal edges for an agent are: ", end="")
    print(*optimal_edges, sep=", ")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
