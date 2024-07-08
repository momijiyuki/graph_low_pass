import inspect
import numpy as np
import os
from sklearn import datasets
# from sklearn.model_selection import train_test_split


def to_abs_path(fpath):
    dirpath = get_caller_path()
    return os.path.normpath(os.path.join(dirpath, fpath))


def get_caller_path() -> str:
    """関数呼び出しをたどり大元のディレクトリのパスを返す

    Returns
    -------
    str
        _description_
    """
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



def digits():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    return X, y
