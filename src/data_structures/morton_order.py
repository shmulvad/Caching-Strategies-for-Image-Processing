import numpy as np
from itertools import product
from typing import Generator

from helper_funcs import MORTON
from data_structures.caching_data_stucture import CachingDataStructure


# Morton order as a caching data structure
class MortonOrder(CachingDataStructure):
    print_name = MORTON
    type_name = "MortonOrder"

    def __init__(self, picture: np.ndarray = None,
                 shape: tuple = None, cache=None, offset: int = 0):
        assert not (picture is None and shape is None), \
            "You have to set *either* picture or shape"
        assert not (picture is not None and shape is not None), \
            "You can't set *both* picture and shape"
        self.cache = cache
        self.offset = offset
        if picture is not None:
            picture = np.array(picture)
            self.dim = len(picture.shape)
            self.__set_shape__(picture.shape)
            self.__set_part_compact_func__()
            self.__set_vals__(picture)
        else:
            self.dim = len(shape)
            self.__set_shape__(shape)
            self.__set_part_compact_func__()
            self.data = np.zeros(np.prod(shape))

    def __set_part_compact_func__(self) -> None:
        """
        Sets the correct parting and compact functions to use based on the
        dimension of picture/shape
        """
        assert self.dim in range(1, 5), \
            "Only supports 1-, 2-, 3- and 4D pictures"
        part_compact_to_use = [
            (self.identity, self.identity),
            (self.part1by1, self.compact1by1),
            (self.part1by2, self.compact1by2),
            (self.part1by3, self.compact1by3),
        ][self.dim - 1]
        self.part_func = part_compact_to_use[0]
        self.compact_func = part_compact_to_use[1]

    def __set_vals__(self, picture) -> None:
        """Sets the internal data object to values of image."""
        data = np.zeros(np.prod(self.shape), dtype=picture.dtype)
        shape_ranges = (range(n) for n in self.shape)
        for key in product(*shape_ranges):
            data[self.morton_encode(key)] = picture[key]
        self.data = data

    def empty_of_same_shape(self) -> 'MortonOrder':
        """
        Returns a Morton Order object of same shape as this with all zeros
        """
        return MortonOrder(shape=self.shape, cache=self.cache,
                           offset=self.get_next_offset())

    def iter_keys(self) -> Generator[tuple, None, None]:
        """
        Returns a generator that yields tuples of the keys in
        internal linear layout (optimal spatial locality)
        """
        for i in range(np.prod(self.shape)):
            yield self.morton_decode(i)

    def internal_index(self, *args: int) -> int:
        """
        Given a number of coordinates, i.e. (6, 7, 1), it returns 222 or
        the morton encoding of these
        """
        return self.morton_encode(args)

    def part1by3(self, x: int) -> int:
        """Inserts three 0 bits after each of the 8 low bits of x"""
        x &= 0x000000ff                 # x = ---- ---- ---- ---- ---- ---- 7654 3210
        x = (x | x << 12) & 0x000f000f  # x = ---- ---- ---- 7654 ---- ---- ---- 3210
        x = (x | x <<  6) & 0x03030303  # x = ---- --76 ---- --54 ---- --32 ---- --10
        x = (x | x <<  3) & 0x11111111  # x = ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
        return x

    def part1by2(self, x: int) -> int:
        """Inserts two 0 bits after each of the 10 low bits of x"""
        x &= 0x000003ff                   # x = ---- ---- ---- ---- ---- --98 7654 3210
        x = (x | (x << 16)) & 0xff0000ff  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x | (x <<  8)) & 0x0300f00f  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x | (x <<  4)) & 0x030c30c3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x | (x <<  2)) & 0x09249249  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        return x

    def part1by1(self, x: int) -> int:
        """Inserts one 0 bit after each of the 16 low bits of x"""
        x &= 0x0000ffff                  # x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x << 8)) & 0x00ff00ff  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x << 4)) & 0x0f0f0f0f  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x << 2)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x << 1)) & 0x55555555  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        return x

    def morton_encode(self, key: tuple) -> int:
        """
        Given a key of type (x, y, z, etc.) (equal to dimension of encoding)
        it computes the morton encoding of these values
        """
        assert len(key) == self.dim, \
            f"Number of args ({len(key)}) does not match up with internal " + \
            f"dimension ({self.dim})."
        val = 0
        for i in range(self.dim):
            val |= self.part_func(key[i]) << i
        return val

    def compact1by3(self, x: int) -> int:
        """
        Inverse of 'part1by3'. Removes all bits not at indices divisible by 4
        """
        x &= 0x11111111                   # x = ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
        x = (x | (x >>  3)) & 0x03030303  # x = ---- --76 ---- --54 ---- --32 ---- --10
        x = (x | (x >>  6)) & 0x000f000f  # x = ---- ---- ---- 7654 ---- ---- ---- 3210
        x = (x | (x >> 12)) & 0x000000ff  # x = ---- ---- ---- ---- ---- ---- 7654 3210
        return x

    def compact1by2(self, x: int) -> int:
        """
        Inverse of 'part1by2'. Removes all bits not at indices divisible by 3
        """
        x &= 0x09249249                   # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x | (x >>  2)) & 0x030c30c3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x | (x >>  4)) & 0x0300f00f  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x | (x >>  8)) & 0xff0000ff  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x | (x >> 16)) & 0x000003ff  # x = ---- ---- ---- ---- ---- --98 7654 3210
        return x

    def compact1by1(self, x: int) -> int:
        """
        Inverse of 'part1by1'. Removes all bits not at indices divisible by 2
        """
        x &= 0x55555555                  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        x = (x | (x >> 1)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x | (x >> 2)) & 0x0f0f0f0f  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x | (x >> 4)) & 0x00ff00ff  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x | (x >> 8)) & 0x0000ffff  # x = ---- ---- ---- ---- fedc ba98 7654 3210
        return x

    def identity(self, x: int) -> int:
        """Returns the input. Useful for parting and compacting for 1D"""
        return x

    def morton_decode(self, code: int) -> tuple:
        """
        Given an integer value, i.e. 222, it decodes this to i.e. (6, 7, 1) for
        a 3D image or (2, 1, 3, 3) for 4D etc.
        """
        return tuple([self.compact_func(code >> i) for i in range(self.dim)])
