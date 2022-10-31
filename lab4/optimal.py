from argparse import ArgumentParser
import numpy as np
import subprocess
import pathlib

path = pathlib.Path(__file__).parent.resolve()
ppath = pathlib.Path(__file__).parent.parent.resolve()


def build_parser() -> ArgumentParser:
    argparser = ArgumentParser(prog="python optimal.py")
    argparser.add_argument(
        "--path", default="graph2.csv", type=str, help="path to your csv file"
    )
    return argparser


def c2(s: str) -> str:
    data = s.split(",")
    data[-1] = str(2 * int(data[-1]))
    return ",".join(data)


def main(args):
    graph_table = np.genfromtxt(args.path, delimiter=";", dtype="str", skip_header=True)
    n = graph_table.shape[1]
    a = int(graph_table.shape[0] / n)

    graph_table2 = np.full((n + n * a, n + n * a), "nan", dtype=object)
    for i in range(n):
        for j in range(a):
            graph_table2[i, n + i * a + j] = "0"
    for k in range(a):
        for i in range(n):
            for j in range(n):
                graph_table2[n + i * a + k, j] = c2(graph_table[k * n + i, j])
    graph_table2 = np.vstack([np.arange(n + n * a), graph_table2])
    t_file = "temp.csv"
    np.savetxt("{}/{}".format(path, t_file), graph_table2, delimiter=";", fmt="%s")
    subprocess.run(
        [
            "python",
            "{}/{}/{}".format(ppath, "lab3", "optimal.py"),
            "--path",
            "{}/{}".format(path, t_file),
            "--print",
        ]
    )
    file = pathlib.Path("{}/{}".format(path, t_file))
    file.unlink()


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
