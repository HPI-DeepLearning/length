import os
import shutil
import pytest

from length.data_sets import MNIST, FashionMNIST

temp_folder = ".temp"


@pytest.fixture(scope="module")
def cleanup():
    yield cleanup
    print("delete temp folder")
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


def test_mnist_downloading(cleanup):
    data_set = MNIST(10, delay_loading=True)
    data_set.path = temp_folder
    data_set.download_files()

    required_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for file in required_files:
        assert os.path.isfile(os.path.join(temp_folder, file))


def test_fashion_mnist_downloading(cleanup):
    data_set = FashionMNIST(10, delay_loading=True)
    data_set.path = temp_folder
    data_set.download_files()

    required_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for file in required_files:
        assert os.path.isfile(os.path.join(temp_folder, file))
