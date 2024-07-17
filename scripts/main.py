import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.typing import ArrayLike, NDArray
from pygsp import graphs
from typing import Annotated, Callable


import mygraph
from mygraph import load_mnist


Vector = Annotated[NDArray[np.float64], "1D"]
Matrix = Annotated[NDArray[np.float64], "2D"]


def func_h(lamda: float, thred = 5) -> int:
    return 1 if lamda < thred else 0


def to_abs(x:float, y:float, /, *, bias: int|float = 0) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            return bias + (x-y if x-y >= 0 else y-x)
        except Warning as e:
            print(f"{x=}{x.dtype} {y=}{y.dtype}")
            raise Exception(e)


def normalize(x:NDArray) -> NDArray:
    return x / 255


def adjacency_matrix(x, width = None) -> Matrix:
    length = len(x)
    if width is None:
        width = int(np.sqrt(length))
    height = length // width

    a = np.zeros([length, length])
    for i in range(length):
        row, column = divmod(i, width)
        if row != 0:
            a[i, i-width] = a[i-width, i] = to_abs(x[i], x[i-width], bias=1)
        if row != height-1:
            a[i, i+width] = a[i+width, i] = to_abs(x[i], x[i+width], bias=1)
        if column != 0:
            a[i, i-1]     = a[i-1, i]     = to_abs(x[i], x[i-1], bias=1)
        if column !=width-1:
            a[i, i+1]     = a[i+1, i]     = to_abs(x[i], x[i+1], bias=1)
    return a


def degree_matrix(a: Vector) -> Matrix:
    if len(a.shape) != 2:
        assert "not 2 dimentional matrix"

    # diag_list = [len(*np.nonzero(i)) for i in a]
    diag_list = [np.sum(i) for i in a]
    return np.diag(diag_list)


def compute_cl(k:int,
               l:int,
               lambda_max:float,
               func_h: Callable[[float], float]
               ) -> float:
    k_s = k + 1
    cl = 0
    for p in range(1, k_s):
        theta_p = (np.pi *(p - 0.5)) / k_s
        cl += np.cos(l*theta_p)* func_h(lambda_max/2 * (np.cos(theta_p)+1))
    return cl * 2 / k_s


def graph_filter(x, k, h, L, lmax) -> Vector:
    n_dim = len(x)
    tl_list = [x, (2*L/lmax - np.eye(n_dim))@x]
    for i in range(2, k):
        tl_list.append(
            2*(2*L/lmax - np.eye(n_dim))@tl_list[i-1] - tl_list[i-2]
        )
    y = compute_cl(k, 0, lmax, h)/2
    for i in range(1, k):
        y += compute_cl(k, i, lmax, h)*tl_list[i]
    return y


def filter_graph_domain(x: Vector) -> None:
    A = adjacemcy_matrix(x)
    g = graphs.Graph(A)
    g.compute_fourier_basis()
    gft_sig = g.gft(x)
    # fig, ax = plt.subplots()
    # ax.stem(g.e, gft_sig, linefmt="--", basefmt="k-", label="correct")
    # plt.show()
    output = np.diag(
        np.array([func_h(i) for i in g.e])
        )@gft_sig

    plt.imshow(g.igft(output).reshape(int(np.sqrt(x.shape[0])), -1), cmap="gray")
    plt.axis("off")
    plt.show()


def filter_vertex_domain(x: Vector) -> None:
    A = adjacemcy_matrix(x)
    D = degree_matrix(A)
    L = D - A
    lmax = np.linalg.eigvalsh(L)[-1]
    print(f"lambda max: {lmax}")

    y = graph_filter(x, 10, func_h, L, lmax)

    plt.imshow(y.reshape(int(np.sqrt(x.shape[0])), -1), cmap="gray")
    plt.axis("off")
    plt.show()


def fillter_vertex_class(x: Vector) -> None:
    A = adjacency_matrix(x)
    D = degree_matrix(A)
    L = D - A
    graph_filter = mygraph.GraphLowPassFilter(x, func_h, L, kmax=10, threshold=3)
    res = graph_filter.apply_filter(10)
    plt.imshow(res.reshape(int(np.sqrt(x.shape[0])), -1), cmap="gray")
    plt.axis("off")
    plt.show()


def main():
    traindata, *_ = load_mnist.mnist(dtype=np.int16)
    data = normalize(traindata[124])

    filter_graph_domain(data)
    filter_vertex_domain(data)
    fillter_vertex_class(data)


if __name__=="__main__":
    main()
