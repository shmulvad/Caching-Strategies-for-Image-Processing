import pytest
import numpy as np

from data_structures.block_array import BlockArray


class TestBlockArr(object):
    def setup_method(self, f):
        pass

    def test_internal_index_correct_2d(self):
        block_arr = BlockArray(shape=(8, 8), S=2)
        assert block_arr.internal_index(0, 0) == 0
        assert block_arr.internal_index(0, 1) == 1
        assert block_arr.internal_index(1, 1) == 3
        assert block_arr.internal_index(7, 3) == 55
        assert block_arr.internal_index(7, 4) == 58
        assert block_arr.internal_index(7, 7) == 8**2 - 1

    def test_internal_index_invalid_arg_2d(self):
        block_arr = BlockArray(shape=(8, 8))
        with pytest.raises(TypeError) as _:
            block_arr.internal_index(1)

        with pytest.raises(TypeError) as _:
            block_arr.internal_index(1, 1, 1)

        with pytest.raises(TypeError) as _:
            block_arr.internal_index(1, 1, 1, 1)

    def test_internal_index_correct_3d(self):
        block_arr = BlockArray(shape=(8, 8, 8))
        assert block_arr.internal_index(0, 0, 0) == 0
        assert block_arr.internal_index(0, 0, 1) == 1
        assert block_arr.internal_index(1, 1, 1) == 73
        assert block_arr.internal_index(7, 3, 3) == 475
        assert block_arr.internal_index(7, 4, 4) == 484
        assert block_arr.internal_index(7, 7, 7) == 8**3 - 1

    def test_internal_index_invalid_arg_3d(self):
        block_arr = BlockArray(shape=(8, 8, 8))
        with pytest.raises(TypeError) as _:
            block_arr.internal_index(1)

        with pytest.raises(TypeError) as _:
            block_arr.internal_index(1, 1)

        with pytest.raises(TypeError) as _:
            block_arr.internal_index(1, 1, 1, 1)

    def test_setting_to_2d_pic(self):
        vals = np.random.rand(4, 4)
        block_arr = BlockArray(picture=vals)
        assert block_arr.dim == 2
        assert block_arr.shape == vals.shape
        np.testing.assert_almost_equal(vals, block_arr.to_numpy())

    def test_setting_to_3d_pic(self):
        vals = np.random.rand(4, 4, 4)
        block_arr = BlockArray(picture=vals)
        assert block_arr.dim == 3
        assert block_arr.shape == vals.shape
        np.testing.assert_almost_equal(vals, block_arr.to_numpy())

    def test_iter_keys_2d(self):
        block_arr = BlockArray(shape=(4, 4))
        assert list(block_arr.iter_keys()) == [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 0), (3, 1), (3, 2), (3, 3)]
        block_arr = BlockArray(shape=(32, 32))
        assert len(list(block_arr.iter_keys())) == 32**2

    def test_iter_keys_3d(self):
        block_arr = BlockArray(shape=(2, 2, 2))
        assert list(block_arr.iter_keys()) == [
            (0, 0, 0), (0, 0, 1),
            (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1),
            (1, 1, 0), (1, 1, 1)
        ]
        block_arr = BlockArray(shape=(8, 8, 8))
        assert len(list(block_arr.iter_keys())) == 8**3

    def test_valid_index_2d(self):
        block_arr = BlockArray(shape=(16, 16))
        assert block_arr.valid_index(0, 0)
        assert block_arr.valid_index(8, 7)
        assert block_arr.valid_index(15, 15)
        assert block_arr.valid_index(1, 1, pad=1)
        assert block_arr.valid_index(14, 14, pad=1)

    def test_invalid_index_2d(self):
        block_arr = BlockArray(shape=(16, 16))
        assert not block_arr.valid_index(5)
        assert not block_arr.valid_index(5, 5, 5)
        assert not block_arr.valid_index(-1, -1)
        assert not block_arr.valid_index(16, 16)
        assert not block_arr.valid_index(0, 0, pad=1)
        assert not block_arr.valid_index(15, 15, pad=1)

    def test_valid_index_3d(self):
        block_arr = BlockArray(shape=(16, 16, 16))
        assert block_arr.valid_index(0, 0, 0)
        assert block_arr.valid_index(8, 7, 4)
        assert block_arr.valid_index(15, 15, 15)
        assert block_arr.valid_index(1, 1, 1, pad=1)
        assert block_arr.valid_index(14, 14, 14, pad=1)

    def test_invalid_index_3d(self):
        block_arr = BlockArray(shape=(16, 16, 16))
        assert not block_arr.valid_index(5)
        assert not block_arr.valid_index(5, 5)
        assert not block_arr.valid_index(5, 5, 5, 5)
        assert not block_arr.valid_index(-1, -1, -1)
        assert not block_arr.valid_index(16, 16, 16)
        assert not block_arr.valid_index(0, 0, 0, pad=1)
        assert not block_arr.valid_index(15, 15, 15, pad=1)

    def test_fill(self):
        block_arr = BlockArray([[False, False], [False, False]])
        for key in block_arr.iter_keys():
            assert not block_arr[key]
        block_arr.fill(True, dtype=bool)
        for key in block_arr.iter_keys():
            assert block_arr[key]

    def test_to_numpy(self):
        vals = np.random.rand(8, 8)
        block_arr = BlockArray(vals)
        np.testing.assert_almost_equal(vals, block_arr.to_numpy())

        vals = np.random.rand(8, 8, 8)
        block_arr = BlockArray(vals)
        np.testing.assert_almost_equal(vals, block_arr.to_numpy())

    def test_empty_of_same_shape(self):
        vals = np.random.rand(8, 8)
        block_arr1 = BlockArray(vals)
        block_arr2 = block_arr1.empty_of_same_shape()  # All 0's in same shape
        assert block_arr1.shape == block_arr2.shape
        np.testing.assert_almost_equal(np.zeros((8, 8)), block_arr2.to_numpy())
