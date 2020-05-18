import numpy as np
from data_structures.row_major_array import RowMajorArray
from algorithms.fft import tuple_gen, rev_bit_str, bit_rev_copy, fftn
from scipy.fft import fftn as fftn_scipy


def rand_complex_array(*args):
    return np.random.rand(*args) + 1j * np.random.rand(*args)


class TestFastFourierTransform(object):
    def setup_method(self, f):
        np.random.seed(0)

    def test_tuple_gen_1d(self):
        assert tuple_gen(5, (0,), 0) == (5,)

    def test_tuple_gen_2d(self):
        assert tuple_gen(5, (0, 0), 0) == (5, 0)
        assert tuple_gen(5, (0, 0), 1) == (0, 5)

    def test_tuple_gen_3d(self):
        assert tuple_gen(5, (0, 0, 0), 0) == (5, 0, 0)
        assert tuple_gen(5, (0, 0, 0), 1) == (0, 5, 0)
        assert tuple_gen(5, (0, 0, 0), 2) == (0, 0, 5)

    def test_rev_bit_str(self):
        assert rev_bit_str(3, 2) == 3     # 11 => 11 => 3
        assert rev_bit_str(0, 4) == 0     # 0000 => 0000 => 0
        assert rev_bit_str(3, 4) == 12    # 0011 => 1100 => 12
        assert rev_bit_str(6, 4) == 6     # 0110 => 0110 => 6
        assert rev_bit_str(12, 4) == 3    # 1100 => 0011 => 3
        assert rev_bit_str(1, 10) == 512  # 0000000001 => 1000000000 => 512

    def test_bit_rev_copy(self):
        arr = [0, 1, 2, 3, 4, 5, 6, 7]
        expected = np.array([0, 4, 2, 6, 1, 5, 3, 7])
        res = bit_rev_copy(RowMajorArray(arr))
        np.testing.assert_almost_equal(expected, res.to_numpy())

    def test_1d(self):
        arr = rand_complex_array(4)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())

        arr = rand_complex_array(2048)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())

    def test_2d(self):
        arr = rand_complex_array(4, 4)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())

        arr = rand_complex_array(64, 64)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())

    def test_3d(self):
        arr = rand_complex_array(4, 4, 4)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())

        arr = rand_complex_array(16, 16, 16)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())

    def test_4d(self):
        arr = rand_complex_array(4, 4, 4, 4)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())

    def test_5d(self):
        arr = rand_complex_array(2, 2, 2, 2, 2)
        res = fftn(RowMajorArray(arr))
        res_scipy = fftn_scipy(arr)
        np.testing.assert_almost_equal(res_scipy, res.to_numpy())
