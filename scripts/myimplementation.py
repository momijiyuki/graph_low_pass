import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.typing import NDArray
from typing import Annotated, Callable

import load_mnist


Vector = Annotated[NDArray[np.float64], "1D"]
Matrix = Annotated[NDArray[np.float64], "2D"]


def to_abs(x, y, /, *, bias = 0):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            return bias + (x-y if x-y >= 0 else y-x)
        except Warning as e:
            print(f"{x=}{x.dtype} {y=}{y.dtype}")
            raise Exception(e)


def normalize(x:np.ndarray) -> np.ndarray:
    return x / 255


def adjacemcy_matrix(x, width = None) -> np.ndarray:
    length = len(x)
    if width is None:
        width = int(np.sqrt(length))
    height = length // width

    a = np.zeros([length, length])

    for i in range(length):
        row, column = divmod(i, width)

        if row != 0:         a[i, i-width] = a[i-width, i] = to_abs(x[i], x[i-width], bias=1)
        if row != height-1:  a[i, i+width] = a[i+width, i] = to_abs(x[i], x[i+width], bias=1)
        if column != 0:      a[i, i-1]     = a[i-1, i]     = to_abs(x[i], x[i-1], bias=1)
        if column !=width-1: a[i, i+1]     = a[i+1, i]     = to_abs(x[i], x[i+1], bias=1)
    return a


def degree_matrix(a:np.ndarray) -> np.ndarray:
    if len(a.shape) != 2:
        assert "not 2 dimentional matrix"

    diag_list = [np.sum(i) for i in a]
    return np.diag(diag_list)


class GraphLowPassFilter:
    def __init__(self,
                x: Vector,
                filter: Callable[[float], float],
                L: Matrix,
                kmax: int = 100
                ) -> None:
        self._x = x
        self.filter = filter
        self._L = L
        self.lmax = np.linalg.eigvalsh(L)[-1]
        self._kmax = kmax
        self._cl_list = self._compute_cl()
        self._tl_list = self._compute_tl()
        print(f"lambda max = {self.lmax}")

    def apply_filter(self, x, k):
        y = 0
        for i in range(k):
            y += self._cl_list[i] * self._tl_list[i]
        return y

    def filter_response(self, k):
        lamda = np.linspace(0, self.lmax, 100)

        tl_list = [1, lamda]
        for i in range(2, self._kmax):
            tl_list.append(
                2*(2*lamda/self.lmax - 1)*tl_list[i-1] - tl_list[i-2]
                )
        y = self._integrate_cl(0)/2
        y = 0
        for i in range(1, k):
            y += self._integrate_cl(i) * tl_list[i]
        return lamda, y

    def _compute_cl(self) -> dict[float]:
        cl_list = [self._integrate_cl(0)/2]

        for i in range(1, self._kmax):
            cl_list.append(self._integrate_cl(i))
        return cl_list

    def _integrate_cl(self, l):

        # def f(theta):
        #     return np.cos(l*theta) * self.filter(
        #         self.lmax/2 * (np.cos(theta + 1)))
        # a = 0
        # b = self.lmax
        # n = 100
        # h = (b-a)/n

        # return h/2 * (f(a)+ f(b) + 2* np.sum([f(a+h*i) for i in range(1, n) ]))


        k_s = self._kmax + 1
        k_s = 10+1
        cl = 0
        for p in range(1, k_s):
            theta_p = (np.pi)/k_s * (p-0.5)
            cl += np.cos(l*theta_p) * self.filter(
                self.lmax/2 * (np.cos(theta_p) + 1)
                )
        return 2/k_s * cl

    def _compute_tl(self):
        n_dim = len(self._x)
        tl_list = [self._x, self._L @ self._x]
        for i in range(2, self._kmax):
            tl_list.append(
                2*(self._L/self.lmax - np.eye(n_dim))@tl_list[i-1] - tl_list[i-2]
                )
        return tl_list


def funch(lamda):
    return 1 if lamda < 5 else 0


def main():
    traindata, *_ = load_mnist.mnist(dtype=np.int16)

    data = normalize(traindata[1])
    A = adjacemcy_matrix(data)
    D = degree_matrix(A)
    L = D - A
    graph_filter = GraphLowPassFilter(data, funch, L, kmax=50)
    # fig, ax = plt.subplots(1, 5)
    x = np.linspace(0, np.linalg.eigvalsh(L)[-1], 1000)
    plt.plot(x, list(funch(i) for i in x), label="base filter")
    for i, j in enumerate(range(10, 51, 10)):

        res = graph_filter.apply_filter(data, j)
        x, y = graph_filter.filter_response(j)
        plt.plot(x, y, label=f"k={j}")
        # ax[i].imshow(res.reshape(int(np.sqrt(data.shape[0])), -1), cmap="gray")
    # plt.imshow(res.reshape(int(np.sqrt(data.shape[0])), -1), cmap="gray")
    # plt.ylim(-1, 2)
    # plt.axis("off")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
