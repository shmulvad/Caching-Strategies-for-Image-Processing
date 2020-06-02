import numpy as np
from itertools import product
from typing import Generator, Callable
from cachesim import CacheSimulator

from helper_funcs import MORTON
from data_structures.caching_data_stucture import CachingDataStructure


# Morton order as a caching data structure
class MortonOrder(CachingDataStructure):
    print_name = MORTON
    type_name = "MortonOrder"

    def __init__(self, picture: np.ndarray = None, shape: tuple = None,
                 cache: CacheSimulator = None, offset: int = 0):
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
            if isinstance(shape, int):
                shape = (shape,)
            self.dim = len(shape)
            self.__set_shape__(shape)
            self.__set_part_compact_func__()
            self.data = np.zeros(np.prod(shape))

    def __set_part_compact_func__(self) -> None:
        """
        Sets the correct parting and compact functions to use based on the
        dimension of picture/shape
        """
        if self.dim <= 4:  # Hardcoded case
            part_compact_to_use = [
                (self.identity, self.identity),
                (self.part1by1, self.compact1by1),
                (self.part1by2, self.compact1by2),
                (self.part1by3, self.compact1by3),
            ][self.dim - 1]
            self.part_func = part_compact_to_use[0]
            self.compact_func = part_compact_to_use[1]
        else:  # Dynamic case
            self.part_func = self.part1byN(self.dim - 1)
            self.compact_func = self.compact1byN(self.dim - 1)

    def __get_hex_const__(self, successive_digits: int, spacing: int,
                          num_bits: int = 32) -> int:
        """
        Returns the hexadecimal constant to be used for bit parting for a given
        number of succesive digits and spacing inbetween

        >>> bin(get_hex_const(2, 3, 16))
        '0b1000110001100011'
        >>> bin(get_hex_const(1, 4, 16))
        '0b1000010000100001'
        """
        hex_num, bit_idx = 0, 0
        while bit_idx < num_bits:
            for _ in range(successive_digits):
                hex_num += 2**bit_idx
                bit_idx += 1
                if bit_idx >= num_bits:
                    break
            bit_idx += spacing
        return hex_num

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

    def part1byN(self, n: int, num_bits: int = 32) -> Callable[[int], int]:
        """Returns a partbyN function for a given n and number of bits"""
        # num_iters should idealy be set more carefully to avoid
        # unneeded computing
        num_iters = 5 * (num_bits // 32)
        ns = [n * 2**i for i in range(num_iters)]
        hex_consts = [self.__get_hex_const__(2**i, ns[i], num_bits)
                      for i in range(num_iters)]

        def part1byN_func(x: int) -> int:
            x &= hex_consts[-1]
            for i in range(num_iters - 2, -1, -1):
                x = (x | (x << ns[i])) & hex_consts[i]
            return x

        return part1byN_func

    def part1by3(self, x: int) -> int:
        """Inserts three 0 bits after each of the 8 low bits of x"""
        x &= 0x000000ff                   # x = ---- ---- ---- ---- ---- ---- 7654 3210
        x = (x | (x << 12)) & 0x000f000f  # x = ---- ---- ---- 7654 ---- ---- ---- 3210
        x = (x | (x <<  6)) & 0x03030303  # x = ---- --76 ---- --54 ---- --32 ---- --10
        x = (x | (x <<  3)) & 0x11111111  # x = ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
        return x

    def part1by2(self, x: int) -> int:
        """Inserts two 0 bits after each of the 11 low bits of x"""
        x &= 0x000007ff                   # x = ---- ---- ---- ---- ---- -a98 7654 3210
        x = (x | (x << 16)) & 0x070000ff  # x = ---- -a98 ---- ---- ---- ---- 7654 3210
        x = (x | (x <<  8)) & 0x0700f00f  # x = ---- -a98 ---- ---- 7654 ---- ---- 3210
        x = (x | (x <<  4)) & 0x430c30c3  # x = -a-- --98 ---- 76-- --54 ---- 32-- --10
        x = (x | (x <<  2)) & 0x49249249  # x = -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        return x

    def part1by1(self, x: int) -> int:
        """Inserts one 0 bit after each of the 16 low bits of x"""
        x &= 0x0000ffff                  # x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x | (x << 8)) & 0x00ff00ff  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x | (x << 4)) & 0x0f0f0f0f  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x | (x << 2)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x | (x << 1)) & 0x55555555  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
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

    def compact1byN(self, n: int, num_bits: int = 32) -> Callable[[int], int]:
        """Returns a compact1byN function for a given n and number of bits"""
        # num_iters should idealy be set more carefully to avoid
        # unneeded computation
        num_iters = 5 * (num_bits // 32)
        ns = [n * 2**i for i in range(num_iters)]
        hexs = [self.__get_hex_const__(2**i, ns[i], num_bits)
                for i in range(num_iters)]

        def compact1byN_func(x: int) -> int:
            x &= hexs[0]
            for i in range(1, num_iters):
                x = (x | (x >> ns[i-1])) & hexs[i]
            return x

        return compact1byN_func

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
