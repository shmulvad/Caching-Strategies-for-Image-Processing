import numpy as np
from scipy import ndimage
import pytest

from data_structures.row_major_array import RowMajorArray
from algorithms.convolution import convolution


class TestConvolution(object):
    kernel_2d = None
    kernel_3d = None

    def setup_method(self, f):
        self.kernel_2d = np.ones((3, 3)) / 3**2.0
        self.kernel_3d = np.ones((3, 3, 3)) / 3**3.0

    def test_error_when_unsupported_dim(self):
        arr = RowMajorArray(np.random.rand(4, 4, 4, 4))
        with pytest.raises(AssertionError) as err:
            convolution(arr, np.ones((3, 3, 3, 3)))
        msg = "Convolution for pictures of dim 4 not implemented"
        assert err.value.args[0] == msg

        arr = RowMajorArray(np.random.rand(4, 4, 4, 4, 4))
        with pytest.raises(AssertionError) as err:
            convolution(arr, np.ones((3, 3, 3, 3, 3)))
        msg = "Convolution for pictures of dim 5 not implemented"
        assert err.value.args[0] == msg

    def test_same_as_scipy_2d(self):
        a = np.ones((16, 16))
        sci = ndimage.convolve(a, self.kernel_2d, mode='constant', cval=3.0)
        own = convolution(RowMajorArray(a), self.kernel_2d, cval=3.0)
        np.testing.assert_almost_equal(sci, own.to_numpy())

    def test_same_as_scipy_3d(self):
        a = np.ones((8, 8, 8))
        sci = ndimage.convolve(a, self.kernel_3d, mode='constant', cval=3.0)
        own = convolution(RowMajorArray(a), self.kernel_3d, cval=3.0)
        np.testing.assert_almost_equal(sci, own.to_numpy())
