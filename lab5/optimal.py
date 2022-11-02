from argparse import ArgumentParser
from networkx import DiGraph
import numpy as np
from scipy.optimize import linprog

from prettytable import PrettyTable


def build_parser() -> ArgumentParser:
    argparser = ArgumentParser(prog="python optimal.py")
    argparser.add_argument("--x0", default=0, type=int, help="node to start")
    argparser.add_argument("--gamma", default="0.5", type=float, help="gamma discount")
    argparser.add_argument(
        "--path", default="graph2.csv", type=str, help="path to your csv file"
    )
    argparser.add_argument(
        "--print", help="print cost table", required=False, action="store_true"
    )
    return argparser


def create_graph(
    graph_table: np.array, graph: DiGraph, z_edges: np.ndarray
) -> np.ndarray:
    n_edge = 0
    n_node = 0
    for i in range(graph_table.shape[0]):
        for j in range(graph_table.shape[1]):
            if graph_table[i, j] != "nan":
                data = graph_table[i, j]
                if "," in data:
                    p, c = data.split(",")
                    graph.add_edge(i, j, p=float(p), c=float(c))
                    graph.nodes[i]["control"] = False
                    if "n" not in graph.nodes[i]:
                        graph.nodes[i]["n"] = n_node
                        n_node += 1
                else:
                    graph.add_edge(i, j, c=float(data), n=n_edge)
                    z_edges = np.vstack([z_edges, np.array([i, j])])
                    graph.nodes[i]["control"] = True
                    n_edge += 1
                    if "n" not in graph.nodes[i]:
                        graph.nodes[i]["n"] = n_node
                        n_node += 1
    return z_edges


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
    z_edges = np.array([]).reshape(0, 2)
    z_edges = create_graph(graph_table[1:], graph, z_edges)
    n_control, n_control_edges = get_n_control_edges(graph)
    n_not_control = graph.number_of_nodes() - n_control
    n = n_control + n_not_control
    gamma = args.gamma
    x0 = args.x0

    z = np.zeros((n_control_edges + n))
    a_eq = np.zeros((graph.number_of_nodes() + 1, n_control_edges + n_not_control))
    a_eq[-1] = np.ones((1, n_control_edges + n_not_control))
    b_eq = np.zeros(n + n_control)
    b_eq[x0] = 1

    a_eq = np.zeros((n + n_control, n_control_edges + n))
    for node in sorted(graph.nodes()):
        for edge in graph.in_edges(node, data=True):
            if graph.nodes[edge[0]]["control"]:
                a_eq[node, edge[2]["n"]] = -gamma
            else:
                a_eq[node, n_control_edges + graph.nodes[edge[0]]["n"]] = (
                    -gamma * edge[2]["p"]
                )
        a_eq[node, n_control_edges + graph.nodes[node]["n"]] = 1
        if graph.nodes[node]["control"]:
            for neighbor in sorted(graph.neighbors(node)):
                z[graph[node][neighbor]["n"]] += graph[node][neighbor]["c"]
                a_eq[n + graph.nodes[node]["n"], graph.edges[node, neighbor]["n"]] = 1
                a_eq[
                    n + graph.nodes[node]["n"], n_control_edges + graph.nodes[node]["n"]
                ] = -1
        else:
            mu = 0
            for neighbor in sorted(graph.neighbors(node)):
                mu += graph[node][neighbor]["p"] * graph[node][neighbor]["c"]
            z[n_control_edges + graph.nodes[node]["n"]] = mu

    result = linprog(z, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")

    optimal_edges = z_edges[np.nonzero(result.x[:n_control_edges])]

    table = PrettyTable()
    table.field_names = np.array(["cost", "from node", "to node"])
    table.add_rows(
        np.append(
            np.expand_dims(np.round(result["x"][:n_control_edges], 3), axis=1),
            z_edges,
            axis=1,
        )
    )
    if args.print:
        print(table)
    print(
        "The value of the objective function in the optimal solution is {}".format(
            np.round(result["fun"], 3)
        )
    )
    print("Optimal edges for an agent are: ", end="")
    print(*optimal_edges, sep=", ")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
