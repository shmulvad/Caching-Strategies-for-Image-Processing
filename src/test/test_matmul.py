import numpy as np
import pytest

from data_structures.row_major_array import RowMajorArray
from algorithms.matmul import matmul_rec


class TestMatmul(object):
    def setup_method(self, f):
        np.random.seed(0)

    def test_error_when_unsupported_dim(self):
        arr1 = RowMajorArray(np.random.rand(4, 4, 4))
        arr2 = RowMajorArray(np.random.rand(4, 4, 4))
        with pytest.raises(AssertionError) as err:
            matmul_rec(arr1, arr2, 10)
        msg = "Arrays should be 2D but got dimensions 3 and 3"
        assert err.value.args[0] == msg

    def test_error_when_not_same_shape(self):
        arr1 = RowMajorArray(np.random.rand(4, 4))
        arr2 = RowMajorArray(np.random.rand(8, 8))
        with pytest.raises(AssertionError) as err:
            matmul_rec(arr1, arr2, 10)
        msg = "Arrays should be same shape but got (4, 4) and (8, 8)"
        assert err.value.args[0] == msg

    def test_same_as_np_basic(self):
        arr1 = np.random.rand(16, 16)
        arr2 = np.random.rand(16, 16)
        res_np = arr1 @ arr2
        # Will use simple matmul straight away when rec depth is 0
        res = matmul_rec(RowMajorArray(arr1), RowMajorArray(arr2), 0)
        np.testing.assert_almost_equal(res_np, res.to_numpy())

    def test_same_as_np_depth(self):
        # Just set depth high so we know we will get to max depth
        r = 100
        arr1 = np.random.rand(4, 4)
        arr2 = np.random.rand(4, 4)
        res_np = arr1 @ arr2
        res = matmul_rec(RowMajorArray(arr1), RowMajorArray(arr2), r)
        np.testing.assert_almost_equal(res_np, res.to_numpy())

        arr1 = np.random.rand(16, 16)
        arr2 = np.random.rand(16, 16)
        res_np = arr1 @ arr2
        res = matmul_rec(RowMajorArray(arr1), RowMajorArray(arr2), r)
        np.testing.assert_almost_equal(res_np, res.to_numpy())

        arr1 = np.random.rand(32, 32)
        arr2 = np.random.rand(32, 32)
        res_np = arr1 @ arr2
        res = matmul_rec(RowMajorArray(arr1), RowMajorArray(arr2), r)
        np.testing.assert_almost_equal(res_np, res.to_numpy())
