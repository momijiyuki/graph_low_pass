import inspect
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from pygsp import graphs


def to_abs_path(fpath):
    dirpath = get_caller_path()
    return os.path.normpath(os.path.join(dirpath, fpath))


def get_caller_path() -> str:
    caller_frame = inspect.stack()[-1]
    caller_module = inspect.getmodule(caller_frame[0])

    caller_module_path = os.path.abspath(caller_module.__file__)
    return os.path.dirname(caller_module_path)


def mnist(datapath = './data', dtype = None):
    """import mnist dataset from PyTorch

    Parameters
    ----------
    datapath : str, optional
        Description of download folder path, by default './data'

    Returns
    -------
    traindata : ndarray
    testdata : ndarray
    trainlabel : ndarray
    testlabel : ndarray

    Examples
    --------
    >>> traindata, testdata, trainlabel, testlabel = mnist_from_torch()
    >>> assert traindata.shape == (60000, 784)
    >>> assert testdata.shape == (10000, 784)
    >>> assert trainlabel.shape == (60000,)
    >>> assert testlabel.shape == (10000,)
    """
    from torchvision.datasets import MNIST

    trainset = MNIST(to_abs_path(datapath), train=True, download=True)
    testset = MNIST(to_abs_path(datapath), train=False, download=True)
    traindata:np.ndarray = trainset.data.detach().numpy().copy().reshape(-1, 784)
    testdata: np.ndarray = testset.data.detach().numpy().copy().reshape(-1, 784)

    trainlabel:np.ndarray = trainset.targets.detach().numpy().copy()
    testlabel :np.ndarray = testset.targets.detach().numpy().copy()

    if dtype is not None:
        return (
            traindata.astype(dtype),
            testdata.astype(dtype),
            trainlabel.astype(dtype),
            testlabel.astype(dtype))

    return traindata, testdata, trainlabel, testlabel


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
        if row != 0:
            a[i, i-width] = a[i-width, i] = to_abs(x[i], x[i-width], bias=1)
        if row != height-1:
            a[i, i+width] = a[i+width, i] = to_abs(x[i], x[i+width], bias=1)
        if column != 0:
            a[i, i-1]     = a[i-1, i]     = to_abs(x[i], x[i-1], bias=1)
        if column !=width-1:
            a[i, i+1]     = a[i+1, i]     = to_abs(x[i], x[i+1], bias=1)
    return a


def func_h(lamda, threshold=5):
    return 1 if lamda < threshold else 0


def save_images(eig_val):
    x = np.linspace(0, eig_val[-1], 1000)
    plt.plot(x, [func_h(i) for i in x])
    plt.xlabel("$\lambda$")
    plt.ylabel(r"$\hat{h}(\lambda)$")
    plt.savefig("filter_response.pdf")
    plt.close()



def main():
    traindata, *_ = mnist(dtype=np.int16)
    data = normalize(traindata[123])
    A = adjacemcy_matrix(data)

    g = graphs.Graph(A)
    g.compute_fourier_basis()
    gft_sig = g.gft(data)
    save_images(g.e)

    fig, ax = plt.subplots()
    ax.stem(g.e, gft_sig, linefmt="--", basefmt="k-", label="correct")
    # plt.show()
    plt.savefig("spectrum.pdf")
    plt.close()

    plt.imshow(data.reshape(int(np.sqrt(data.shape[0])), -1), cmap="gray")
    plt.axis("off")
    plt.savefig("base_image.pdf")
    plt.close()

    fig, ax = plt.subplots(1, 5)
    for i in range(1, 6):
        output = np.diag(
            np.array([func_h(j, i) for j in g.e])
            )@gft_sig
        ax[i-1].imshow(g.igft(output).reshape(int(np.sqrt(data.shape[0])), -1), cmap="gray")
        ax[i-1].axis("off")
        # ax[i-1].set_title(f" $\lambda={i}$"+"以降を遮断 ")
        ax[i-1].set_title(f"$\lambda={i}$")
    plt.tight_layout()
    # plt.show()
    plt.savefig("output.pdf")


if __name__=="__main__":
    main()
