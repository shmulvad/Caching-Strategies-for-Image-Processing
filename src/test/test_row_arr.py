import pytest
import numpy as np

from data_structures.row_major_array import RowMajorArray


class TestRowMajorArray(object):
    def setup_method(self, f):
        pass

    def test_internal_index_correct_2d(self):
        row_arr = RowMajorArray(shape=(8, 8))
        assert row_arr.internal_index(0, 0) == 0
        assert row_arr.internal_index(0, 1) == 1
        assert row_arr.internal_index(1, 1) == 8 + 1
        assert row_arr.internal_index(7, 3) == 7 * 8 + 3
        assert row_arr.internal_index(7, 4) == 7 * 8 + 4
        assert row_arr.internal_index(7, 7) == 8**2 - 1

    def test_internal_index_invalid_arg_2d(self):
        row_arr = RowMajorArray(shape=(8, 8))
        with pytest.raises(TypeError) as _:
            row_arr.internal_index(1)

        with pytest.raises(TypeError) as _:
            row_arr.internal_index(1, 1, 1)

        with pytest.raises(TypeError) as _:
            row_arr.internal_index(1, 1, 1, 1)

    def test_internal_index_correct_3d(self):
        row_arr = RowMajorArray(shape=(8, 8, 8))
        assert row_arr.internal_index(0, 0, 0) == 0
        assert row_arr.internal_index(0, 0, 1) == 1
        assert row_arr.internal_index(1, 1, 1) == 8**2 + 8 + 1
        assert row_arr.internal_index(7, 3, 3) == 7 * 8**2 + 3 * 8 + 3
        assert row_arr.internal_index(7, 4, 4) == 7 * 8**2 + 4 * 8 + 4
        assert row_arr.internal_index(7, 7, 7) == 8**3 - 1

    def test_internal_index_invalid_arg_3d(self):
        row_arr = RowMajorArray(shape=(8, 8, 8))
        with pytest.raises(TypeError) as _:
            row_arr.internal_index(1)

        with pytest.raises(TypeError) as _:
            row_arr.internal_index(1, 1)

        with pytest.raises(TypeError) as _:
            row_arr.internal_index(1, 1, 1, 1)

    def test_setting_to_2d_pic(self):
        vals = np.random.rand(4, 4)
        row_arr = RowMajorArray(picture=vals)
        assert row_arr.dim == 2
        assert row_arr.shape == vals.shape
        np.testing.assert_almost_equal(vals, row_arr.to_numpy())

    def test_setting_to_3d_pic(self):
        vals = np.random.rand(4, 4, 4)
        row_arr = RowMajorArray(picture=vals)
        assert row_arr.dim == 3
        assert row_arr.shape == vals.shape
        np.testing.assert_almost_equal(vals, row_arr.to_numpy())

    def test_iter_keys_2d(self):
        row_arr = RowMajorArray(shape=(4, 4))
        assert list(row_arr.iter_keys()) == [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 0), (3, 1), (3, 2), (3, 3)
        ]
        row_arr = RowMajorArray(shape=(32, 32))
        assert len(list(row_arr.iter_keys())) == 32**2

    def test_iter_keys_3d(self):
        row_arr = RowMajorArray(shape=(2, 2, 2))
        assert list(row_arr.iter_keys()) == [
            (0, 0, 0), (0, 0, 1),
            (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1),
            (1, 1, 0), (1, 1, 1)]
        row_arr = RowMajorArray(shape=(8, 8, 8))
        assert len(list(row_arr.iter_keys())) == 8**3

    def test_valid_index_2d(self):
        row_arr = RowMajorArray(shape=(16, 16))
        assert row_arr.valid_index(0, 0)
        assert row_arr.valid_index(8, 7)
        assert row_arr.valid_index(15, 15)
        assert row_arr.valid_index(1, 1, pad=1)
        assert row_arr.valid_index(14, 14, pad=1)

    def test_invalid_index_2d(self):
        row_arr = RowMajorArray(shape=(16, 16))
        assert not row_arr.valid_index(5)
        assert not row_arr.valid_index(5, 5, 5)
        assert not row_arr.valid_index(-1, -1)
        assert not row_arr.valid_index(16, 16)
        assert not row_arr.valid_index(0, 0, pad=1)
        assert not row_arr.valid_index(15, 15, pad=1)

    def test_valid_index_3d(self):
        row_arr = RowMajorArray(shape=(16, 16, 16))
        assert row_arr.valid_index(0, 0, 0)
        assert row_arr.valid_index(8, 7, 4)
        assert row_arr.valid_index(15, 15, 15)
        assert row_arr.valid_index(1, 1, 1, pad=1)
        assert row_arr.valid_index(14, 14, 14, pad=1)

    def test_invalid_index_3d(self):
        row_arr = RowMajorArray(shape=(16, 16, 16))
        assert not row_arr.valid_index(5)
        assert not row_arr.valid_index(5, 5)
        assert not row_arr.valid_index(5, 5, 5, 5)
        assert not row_arr.valid_index(-1, -1, -1)
        assert not row_arr.valid_index(16, 16, 16)
        assert not row_arr.valid_index(0, 0, 0, pad=1)
        assert not row_arr.valid_index(15, 15, 15, pad=1)

    def test_fill(self):
        row_arr = RowMajorArray([[False, False], [False, False]])
        for key in row_arr.iter_keys():
            assert not row_arr[key]
        row_arr.fill(True, dtype=bool)
        for key in row_arr.iter_keys():
            assert row_arr[key]

    def test_to_numpy(self):
        vals = np.random.rand(8, 8)
        row_arr = RowMajorArray(vals)
        np.testing.assert_almost_equal(vals, row_arr.to_numpy())

        vals = np.random.rand(8, 8, 8)
        row_arr = RowMajorArray(vals)
        np.testing.assert_almost_equal(vals, row_arr.to_numpy())

    def test_empty_of_same_shape(self):
        vals = np.random.rand(8, 8)
        row_arr1 = RowMajorArray(vals)
        row_arr2 = row_arr1.empty_of_same_shape()  # All 0's in same shape
        assert row_arr1.shape == row_arr2.shape
        np.testing.assert_almost_equal(np.zeros((8, 8)), row_arr2.to_numpy())
