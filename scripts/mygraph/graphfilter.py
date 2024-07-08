import numpy as np

from numpy.typing import NDArray
from typing import Annotated, Callable


Vector = Annotated[NDArray[np.float64], "1D"]
Matrix = Annotated[NDArray[np.float64], "2D"]


class GraphLowPassFilter:
    def __init__(self,
                x: Vector,
                filter: Callable[[float], float],
                L: Matrix,
                kmax: int = 100,
                threshold = 3
                ) -> None:
        self._x = x
        self.filter = filter
        self._L = L
        self.lmax = np.linalg.eigvalsh(L)[-1]
        self._kmax = kmax
        self._thred = threshold
        self._cl_list = self._compute_cl()
        self._tl_list = self._compute_tl()

    def apply_filter(self, k):
        y = 0
        for i in range(k):
            y += self._cl_list[i] * self._tl_list[i]
        return y

    def filter_response(self, k):
        lamda = np.linspace(0, self.lmax, 100)

        tl_list = [1, 2*lamda/self.lmax-1]
        for i in range(2, self._kmax):
            tl_list.append(
                2*(2*lamda/self.lmax - 1)*tl_list[i-1] - tl_list[i-2]
                )
        # y = self._integrate_cl(0)/2
        y = self._cl_list[0]
        for i in range(1, k):
            y += self._cl_list[i] * tl_list[i]
            # y += self._integrate_cl(i) * tl_list[i]
        return lamda, y

    def _compute_cl(self) -> list[float]:
        cl_list = [self._integrate_cl(0)/2]

        for i in range(1, self._kmax):
            cl_list.append(self._integrate_cl(i))
        return cl_list

    def _integrate_cl(self, l, k = None):
        if k is None:
            k = self._kmax
        k_s = k+1
        cl = 0
        for p in range(1, k_s):
            theta_p = (np.pi)/k_s * (p-0.5)
            cl += np.cos(l*theta_p) * self.filter(
                self.lmax/2 * (np.cos(theta_p) + 1), self._thred
                )
        return 2/k_s * cl

    def _compute_tl(self):
        n_dim = len(self._x)
        tl_list = [self._x, (2*self._L/self.lmax - np.eye(n_dim)) @ self._x]
        for i in range(2, self._kmax):
            tl_list.append(
                2*(2*self._L/self.lmax - np.eye(n_dim))@tl_list[i-1] - tl_list[i-2]
                )
        return tl_list
