from chainer.initializers.uniform import Uniform
from chainer import initializer

import numpy as np


class HeUniformTorch(initializer.Initializer):
    """
    Compute initial parameters with He initialization.
    """

    def __init__(self, a=np.sqrt(5), dtype=None, **kwargs):
        super(HeUniformTorch, self).__init__(dtype)
        self._a = a

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, _ = initializer.get_fans(array)
        gain = self._calculate_gain(self._a)
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std
        Uniform(scale=bound, dtype=self.dtype)(array)

    def _calculate_gain(self, a):
        return np.sqrt(2.0 / (1 + a**2))


class LinearBiasInitializerTorch(initializer.Initializer):
    """
    Initializer same as pytorch's implementation
    """

    def __init__(self, fan_in, dtype=None, **kwargs):
        super(LinearBiasInitializerTorch, self).__init__(dtype)
        self._fan_in = fan_in

    def __call__(self, array):
        bound = 1.0 / np.sqrt(self._fan_in)
        Uniform(scale=bound, dtype=self.dtype)(array)
