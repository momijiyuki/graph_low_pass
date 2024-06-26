import matplotlib.pyplot as plt
import numpy as np
import warnings

from pygsp import graphs

import load_mnist


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


def compute_cl(k, l, lambda_max, func_h):
    k_s = k + 1
    cl = 0
    for p in range(1, k_s):
        theta_p = (np.pi *(p - 0.5)) / k_s
        cl += np.cos(l*theta_p)* func_h(lambda_max/2 * (theta_p+1))
    return cl


def adjacemcy_matrix(x, width = None) -> np.ndarray:
    length = len(x)
    if width is None:
        width = int(np.sqrt(length))
    height = length // width

    a = np.zeros([length, length])

    for i in range(length):
        row, column = divmod(i, width)
        # TODO: 「RuntimeWarning: overflow encountered in scalar subtract」が発生
        # おそらく0の絶対値をとろうとしてるんじゃないか
        # 画像のピクセル間の数値の変化がないのと繋がりが無いのを等価にしていいの？
        if row != 0:         a[i, i-width] = a[i-width, i] = to_abs(x[i], x[i-width], bias=1)
        if row != height-1:  a[i, i+width] = a[i+width, i] = to_abs(x[i], x[i+width], bias=1)
        if column != 0:      a[i, i-1]     = a[i-1, i]     = to_abs(x[i], x[i-1], bias=1)
        if column !=width-1: a[i, i+1]     = a[i+1, i]     = to_abs(x[i], x[i+1], bias=1)
    return a


def degree_matrix(a:np.ndarray) -> np.ndarray:
    if len(a.shape) != 2:
        assert "not 2 dimentional matrix"

    diag_list = [len(*np.nonzero(i)) for i in a]
    return np.diag(diag_list)


def main():
    # traindata, *_ = load_mnist.mnist(dtype=np.int16)
    traindata, _ = load_mnist.digits()
    # traindata = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    A = adjacemcy_matrix(traindata[0])
    D = degree_matrix(A)
    L = D - A
    eigen_val, eigen_vec = np.linalg.eig(L)
    print(eigen_val)
    return
    g = graphs.Graph(A)
    g.compute_fourier_basis()
    gft_sig = g.gft(traindata[0])


    fig, ax = plt.subplots()
    ax.stem(eigen_val, gft_sig, linefmt="--", basefmt="k-", label="correct")


    # plt.imshow(g.igft(gft_sig).reshape(int(np.sqrt(traindata[0].shape[0])), -1), cmap="gray")
    plt.show()
    # print(gft_sig)
    # print(gft_sig.shape)
    # graphs.Graph.gft(g)

if __name__=="__main__":
    main()
