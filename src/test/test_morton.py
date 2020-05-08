import pytest
import numpy as np

from data_structures.morton_order import MortonOrder


class TestMorton(object):
    def setup_method(self, f):
        pass

    def test_part1by1(self):
        morton = MortonOrder(shape=(8, 8))
        assert morton.part1by1(0b0) == 0b0
        assert morton.part1by1(0b101) == 0b010001
        assert morton.part1by1(0b111) == 0b010101
        assert morton.part1by1(0b10101010) == 0b0100010001000100
        assert morton.part1by1(0b11111111) == 0b0101010101010101

    def test_compact1by1(self):
        morton = MortonOrder(shape=(8, 8))
        assert morton.compact1by1(0b0) == 0b0
        assert morton.compact1by1(0b010001) == 0b101
        assert morton.compact1by1(0b010101) == 0b111
        assert morton.compact1by1(0b0100010001000100) == 0b10101010
        assert morton.compact1by1(0b0101010101010101) == 0b11111111

    def test_part1by2(self):
        morton = MortonOrder(shape=(8, 8, 8))
        assert morton.part1by2(0b0) == 0b0
        assert morton.part1by2(0b101) == 0b001000001
        assert morton.part1by2(0b111) == 0b001001001
        assert morton.part1by2(0b101010) == 0b001000001000001000
        assert morton.part1by2(0b111111) == 0b001001001001001001

    def test_compact1by2(self):
        morton = MortonOrder(shape=(8, 8, 8))
        assert morton.compact1by2(0b0) == 0b0
        assert morton.compact1by2(0b001000001) == 0b101
        assert morton.compact1by2(0b001001001) == 0b111
        assert morton.compact1by2(0b001000001000001000) == 0b101010
        assert morton.compact1by2(0b001001001001001001) == 0b111111

    def test_part1by3(self):
        morton = MortonOrder(shape=(8, 8, 8, 8))
        assert morton.part1by3(0b0) == 0b0
        assert morton.part1by3(0b101) == 0b000100000001
        assert morton.part1by3(0b111) == 0b000100010001
        assert morton.part1by3(0b101010) == 0b000100000001000000010000
        assert morton.part1by3(0b111111) == 0b000100010001000100010001

    def test_compact1by3(self):
        morton = MortonOrder(shape=(8, 8, 8, 8))
        assert morton.compact1by3(0b0) == 0b0
        assert morton.compact1by3(0b000100000001) == 0b101
        assert morton.compact1by3(0b000100010001) == 0b111
        assert morton.compact1by3(0b000100000001000000010000) == 0b101010
        assert morton.compact1by3(0b000100010001000100010001) == 0b111111

    def test_internal_index_correct_2d(self):
        morton = MortonOrder(shape=(8, 8))
        assert morton.internal_index(0, 0) == 0
        assert morton.internal_index(0, 1) == 2
        assert morton.internal_index(1, 1) == 3
        assert morton.internal_index(7, 3) == 31
        assert morton.internal_index(7, 4) == 53
        assert morton.internal_index(50, 50) == 3852
        assert morton.internal_index(7, 7) == 8**2 - 1

    def test_internal_index_invalid_arg_2d(self):
        morton = MortonOrder(shape=(8, 8))
        with pytest.raises(AssertionError) as err:
            morton.internal_index(1)
        msg = "Number of args (1) does not match up with internal dimension (2)."
        assert err.value.args[0] == msg

        with pytest.raises(AssertionError) as err:
            morton.internal_index(1, 1, 1)
        msg = "Number of args (3) does not match up with internal dimension (2)."
        assert err.value.args[0] == msg

    def test_internal_index_correct_3d(self):
        morton = MortonOrder(shape=(8, 8, 8))
        assert morton.internal_index(0, 0, 0) == 0
        assert morton.internal_index(0, 0, 1) == 4
        assert morton.internal_index(1, 1, 1) == 7
        assert morton.internal_index(7, 3, 3) == 127
        assert morton.internal_index(7, 4, 4) == 457
        assert morton.internal_index(7, 7, 7) == 8**3 - 1

    def test_internal_index_invalid_arg_3d(self):
        morton = MortonOrder(shape=(8, 8, 8))
        with pytest.raises(AssertionError) as err:
            morton.internal_index(1)
        assert err.value.args[0] == "Number of args (1) does not match up with internal dimension (3)."

        with pytest.raises(AssertionError) as err:
            morton.internal_index(1, 1)
        assert err.value.args[0] == "Number of args (2) does not match up with internal dimension (3)."

    def test_setting_to_2d_pic(self):
        vals = np.random.rand(4, 4)
        morton = MortonOrder(picture=vals)
        assert morton.dim == 2
        assert morton.shape == vals.shape
        np.testing.assert_almost_equal(vals, morton.to_numpy())

    def test_setting_to_3d_pic(self):
        vals = np.random.rand(4, 4, 4)
        morton = MortonOrder(picture=vals)
        assert morton.dim == 3
        assert morton.shape == vals.shape
        np.testing.assert_almost_equal(vals, morton.to_numpy())

    def test_iter_keys_2d(self):
        morton = MortonOrder(shape=(4, 4))
        assert list(morton.iter_keys()) == [
            (0, 0), (1, 0), (0, 1), (1, 1),
            (2, 0), (3, 0), (2, 1), (3, 1),
            (0, 2), (1, 2), (0, 3), (1, 3),
            (2, 2), (3, 2), (2, 3), (3, 3)
        ]
        morton = MortonOrder(shape=(32, 32))
        assert len(list(morton.iter_keys())) == 32**2

    def test_iter_keys_3d(self):
        morton = MortonOrder(shape=(2, 2, 2))
        assert list(morton.iter_keys()) == [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)
        ]
        morton = MortonOrder(shape=(8, 8, 8))
        assert len(list(morton.iter_keys())) == 8**3

    def test_morton_decode_2d(self):
        morton = MortonOrder(shape=(32, 32))
        assert morton.morton_decode(morton.morton_encode((0, 0))) == (0, 0)
        assert morton.morton_decode(morton.morton_encode((5, 5))) == (5, 5)
        assert morton.morton_decode(morton.morton_encode((7, 7))) == (7, 7)
        assert morton.morton_decode(morton.morton_encode((21, 28))) == (21, 28)
        assert morton.morton_decode(morton.morton_encode((21, 28))) != (21, 29)

    def test_morton_decode_3d(self):
        morton = MortonOrder(shape=(16, 16, 16))
        assert morton.morton_decode(morton.morton_encode((0, 0, 0))) == (0, 0, 0)
        assert morton.morton_decode(morton.morton_encode((5, 5, 5))) == (5, 5, 5)
        assert morton.morton_decode(morton.morton_encode((7, 7, 7))) == (7, 7, 7)
        assert morton.morton_decode(morton.morton_encode((14, 14, 14))) == (14, 14, 14)
        assert morton.morton_decode(morton.morton_encode((14, 14, 14))) != (14, 14, 15)

    def test_valid_index_2d(self):
        morton = MortonOrder(shape=(16, 16))
        assert morton.valid_index(0, 0)
        assert morton.valid_index(8, 7)
        assert morton.valid_index(15, 15)
        assert morton.valid_index(1, 1, pad=1)
        assert morton.valid_index(14, 14, pad=1)

    def test_invalid_index_2d(self):
        morton = MortonOrder(shape=(16, 16))
        assert not morton.valid_index(5)
        assert not morton.valid_index(5, 5, 5)
        assert not morton.valid_index(5, 5, 5, 5)
        assert not morton.valid_index(-1, -1)
        assert not morton.valid_index(16, 16)
        assert not morton.valid_index(0, 0, pad=1)
        assert not morton.valid_index(15, 15, pad=1)

    def test_valid_index_3d(self):
        morton = MortonOrder(shape=(16, 16, 16))
        assert morton.valid_index(0, 0, 0)
        assert morton.valid_index(8, 7, 4)
        assert morton.valid_index(15, 15, 15)
        assert morton.valid_index(1, 1, 1, pad=1)
        assert morton.valid_index(14, 14, 14, pad=1)

    def test_invalid_index_3d(self):
        morton = MortonOrder(shape=(16, 16, 16))
        assert not morton.valid_index(5)
        assert not morton.valid_index(5, 5)
        assert not morton.valid_index(5, 5, 5, 5)
        assert not morton.valid_index(-1, -1, -1)
        assert not morton.valid_index(16, 16, 16)
        assert not morton.valid_index(0, 0, 0, pad=1)
        assert not morton.valid_index(15, 15, 15, pad=1)

    def test_fill(self):
        morton = MortonOrder([[False, False], [False, False]])
        for key in morton.iter_keys():
            assert not morton[key]
        morton.fill(True, dtype=bool)
        for key in morton.iter_keys():
            assert morton[key]

    def test_to_numpy(self):
        vals = np.random.rand(8, 8)
        morton = MortonOrder(vals)
        np.testing.assert_almost_equal(vals, morton.to_numpy())

        vals = np.random.rand(8, 8, 8)
        morton = MortonOrder(vals)
        np.testing.assert_almost_equal(vals, morton.to_numpy())

    def test_empty_of_same_shape(self):
        vals = np.random.rand(8, 8)
        morton1 = MortonOrder(vals)
        morton2 = morton1.empty_of_same_shape()  # All 0's in same shape
        assert morton1.shape == morton2.shape
        np.testing.assert_almost_equal(np.zeros((8, 8)), morton2.to_numpy())
