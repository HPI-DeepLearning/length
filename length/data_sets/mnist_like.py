import numpy as np
import os
import gzip
import struct
from urllib.request import urlretrieve

from length.constants import DTYPE
from length.data_set import DataSet, Batch


class MNISTLike(DataSet):
    """
    Class for all data sets which are in the same format as MNIST
    """
    name = None
    url = None

    def __init__(self, batch_size, sample_dimensions=1, load_train=True, load_test=True, delay_loading=False,
                 scale=1.0, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.data_url = self.url
        self.path = os.path.join(".data", self.name)
        self.sample_dimensions = sample_dimensions
        self.scale = scale

        self.load_train = load_train
        self.load_test = load_test

        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

        # store which file should be written to which property
        self.basenames = {}
        if self.load_train:
            self.basenames["train-images-idx3-ubyte.gz"] = "train_images"
            self.basenames["train-labels-idx1-ubyte.gz"] = "train_labels"
        if self.load_test:
            self.basenames["t10k-images-idx3-ubyte.gz"] = "test_images"
            self.basenames["t10k-labels-idx1-ubyte.gz"] = "test_labels"

        if not delay_loading:
            self.prepare()

    @property
    def test(self):
        return self.batch_iterator(self.test_images, self.test_labels)

    @property
    def train(self):
        return self.batch_iterator(self.train_images, self.train_labels)

    def batch_iterator(self, data_array, label_array):
        for i in range(len(label_array) // self.batch_size):
            data = data_array[self.batch_size * i:self.batch_size * (i + 1)]
            labels = label_array[self.batch_size * i:self.batch_size * (i + 1)]
            yield Batch(data, labels)

    def prepare(self):
        self.download_files()
        self.load_files_into_memory()
        self.check_sanity()

    def download_files(self):
        os.makedirs(self.path, exist_ok=True)

        urls = [self.data_url + f for f in self.basenames]
        for url, basename in zip(urls, self.basenames):
            output_file_name = os.path.join(self.path, basename)
            if not os.path.isfile(output_file_name):
                print("downloading file %s..." % basename)
                urlretrieve(url, output_file_name)

    def load_files_into_memory(self):
        for file_name, target in self.basenames.items():
            file_path = os.path.join(self.path, file_name)
            with gzip.open(file_path) as handle:
                # see IDX file format specification:
                # http://yann.lecun.com/exdb/mnist/
                dtypes = {
                    "08": np.uint8,
                    "09": np.int8,
                    "0B": np.int16,
                    "0C": np.int32,
                    "0D": np.float32,
                    "0E": np.float64,
                }

                magic_num = handle.read(4).hex()
                assert "0000" == magic_num[:4]
                dtype = dtypes[magic_num[4:6]]
                num_dim = int(magic_num[6:8], 16)

                dimensions = [int(handle.read(4).hex(), 16) for _ in range(num_dim)]

                data = np.fromstring(handle.read(), dtype=dtype)

                if "images" in target:
                    # only do this if we are reading an image file
                    shape = dimensions
                    if num_dim > 1 + self.sample_dimensions:
                        shape = [dimensions[0]] + [np.prod(dimensions[1:])]
                    elif num_dim < 1 + self.sample_dimensions:
                        shape = [dimensions[0]] + [1] + dimensions[1:]

                    data = data.reshape(shape)

                    if self.scale is not None:
                        # convert data to internally used dtype (float)
                        data = data.astype(DTYPE) / 255 * self.scale

                # set loaded data
                setattr(self, target, data)

    def check_sanity(self):
        if self.load_test:
            assert self.test_images is not None
            assert self.test_labels is not None
            assert len(self.test_labels) == len(self.test_images)

        if self.load_train:
            assert self.train_images is not None
            assert self.train_labels is not None
            assert len(self.train_labels) == len(self.train_images)
