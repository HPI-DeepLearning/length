import numpy as np

from length import constants


def init(array):
    return np.array(array, dtype=constants.DTYPE)


def retry(max_num_retries):
    def retry_wrapper(func):
        def wrapper():
            num_errored = 0
            error = None
            for _ in range(max_num_retries):
                try:
                    func()
                    break
                except Exception as e:
                    num_errored += 1
                    error = e

            if num_errored == max_num_retries:
                raise error

        return wrapper
    return retry_wrapper
