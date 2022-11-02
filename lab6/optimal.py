from argparse import ArgumentParser
import numpy as np
from scipy.optimize import linprog


def build_parser() -> ArgumentParser:
    argparser = ArgumentParser(prog="python optimal.py")
    argparser.add_argument(
        "--gamma", default="0.5", type=float, help="gamma discount"
    )
    argparser.add_argument(
        "--path", default="graph2.csv", type=str, help="path to your csv file"
    )
    return argparser


def main(args):
    graph = np.genfromtxt(args.path, delimiter=";", dtype="str", skip_header=True)
    n = graph.shape[1]
    a = int(graph.shape[0] / n)

    def split(pair):
        return tuple(map(float, pair.split(",")))
    split_vec = np.vectorize(split)
    p, c = split_vec(graph)

    mu = np.sum(p * c, axis=1)
    A_eq = -args.gamma*p.T+np.tile(np.eye(n), a)
    solution = linprog(mu, A_eq=A_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
    edges = np.array(*map(lambda x: (x % n, x//a), np.nonzero(solution.x)))
    print("Optimal edges [n, a] for the agent are: ", *edges.T)


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
