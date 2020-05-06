import numpy as np
from itertools import product

from data_structures.caching_data_stucture import CachingDataStructure


# Morton order as a caching data structure
class MortonOrder(CachingDataStructure):
    def __init__(self, picture=None, shape=None, cache=None, offset=0):
        assert not (picture is None and shape is None), \
            "You have to set *either* picture or shape"
        assert not (picture is not None and shape is not None), \
            "You can't set *both* picture and shape"
        self.cache = cache
        self.offset = offset
        if picture is not None:
            picture = np.array(picture)
            self.dim = len(picture.shape)
            self.shape = picture.shape
            self.__set_part_compact_func__()
            self.__set_vals__(picture)
        else:
            self.dim = len(shape)
            self.shape = shape
            self.__set_part_compact_func__()
            self.data = np.zeros(np.prod(shape))

    def __set_part_compact_func__(self):
        """
        Sets the correct parting and compact functions to use based on the
        dimension of picture/shape
        """
        assert self.dim >= 2 and self.dim <= 4, "Only supports 2-, 3- and 4D pictures"
        func_idx = self.dim - 2
        self.part_func = [self.part1by1, self.part1by2, self.part1by3][func_idx]
        self.compact_func = [self.compact1by1, self.compact1by2, self.compact1by3][func_idx]

    def __set_vals__(self, picture):
        """Sets the internal data object to values of image."""
        data = np.zeros(np.prod(self.shape), dtype=picture.dtype)
        shape_ranges = (range(val) for val in self.shape)
        for key in product(*shape_ranges):
            data[self.morton_encode(key)] = picture[key]
        self.data = data

    def get_next_offset(self):
        """
        Returns the offset that the next array should start at if starting
        directly after this array
        """
        return self.offset + 8 * np.prod(self.shape)

    def empty_of_same_shape(self):
        """
        Returns a Morton Order object of same shape as this with all zeros
        """
        return MortonOrder(shape=self.shape, cache=self.cache,
                           offset=self.get_next_offset())

    def valid_index(self, *args, pad=0):
        """
        Takes a index like valid_index(434, 23, 49) and optinally a
        padding and returns a bool indicating if the index is within dim
        when removing padding
        """
        return len(args) == self.dim and \
            all([args[i] >= pad and args[i] < self.shape[i] - pad
                 for i in range(self.dim)])

    def iter_keys(self):
        """
        Returns a generator that yields a tuple of the keys in
        internal linear layout (optimal spatial locality)
        """
        for i in range(np.prod(self.shape)):
            yield self.morton_decode(i)

    def map(self, f):
        """
        Higher order function that maps a function f to each element in object
        """
        ret_data = self.empty_of_same_shape()
        for key in self.iter_keys():
            val = f(self.__getitem__(key))
            ret_data.__setitem__(key, val)
        return ret_data

    def internal_index(self, *args):
        """
        Given a number of coordinates, i.e. (6, 7, 1), it returns 222 or
        the morton encoding of these
        """
        return self.morton_encode(args)

    def part1by3(self, x):
        """Inserts three 0 bits after each of the 8 low bits of x"""
        x &= 0x000000ff                 # x = ---- ---- ---- ---- ---- ---- 7654 3210
        x = (x | x << 12) & 0x000f000f  # x = ---- ---- ---- 7654 ---- ---- ---- 3210
        x = (x | x <<  6) & 0x03030303  # x = ---- --76 ---- --54 ---- --32 ---- --10
        x = (x | x <<  3) & 0x11111111  # x = ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
        return x

    def part1by2(self, x):
        """Inserts two 0 bits after each of the 10 low bits of x"""
        x &= 0x000003ff                   # x = ---- ---- ---- ---- ---- --98 7654 3210
        x = (x | (x << 16)) & 0xff0000ff  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x | (x <<  8)) & 0x0300f00f  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x | (x <<  4)) & 0x030c30c3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x | (x <<  2)) & 0x09249249  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        return x

    def part1by1(self, x):
        """Inserts one 0 bit after each of the 16 low bits of x"""
        x &= 0x0000ffff                  # x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x << 8)) & 0x00ff00ff  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x << 4)) & 0x0f0f0f0f  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x << 2)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x << 1)) & 0x55555555  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        return x

    def morton_encode(self, key):
        """
        Given a key of type (x, y, z, etc.) (equal to dimension of encoding)
        it computes the morton encoding of these values
        """
        assert len(key) == self.dim, \
            f"Number of args ({len(key)}) does not match up with internal \
                dimension ({self.dim})."
        val = 0
        for i in range(self.dim):
            val |= self.part_func(key[i]) << i
        return val

    def compact1by3(self, x):
        """
        Inverse of 'part1by3'. Removes all bits not at indices divisible by 4
        """
        x &= 0x11111111                   # x = ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
        x = (x | (x >>  3)) & 0x03030303  # x = ---- --76 ---- --54 ---- --32 ---- --10
        x = (x | (x >>  6)) & 0x000f000f  # x = ---- ---- ---- 7654 ---- ---- ---- 3210
        x = (x | (x >> 12)) & 0x000000ff  # x = ---- ---- ---- ---- ---- ---- 7654 3210
        return x

    def compact1by2(self, x):
        """
        Inverse of 'part1by2'. Removes all bits not at indices divisible by 3
        """
        x &= 0x09249249                   # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x | (x >>  2)) & 0x030c30c3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x | (x >>  4)) & 0x0300f00f  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x | (x >>  8)) & 0xff0000ff  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x | (x >> 16)) & 0x000003ff  # x = ---- ---- ---- ---- ---- --98 7654 3210
        return x

    def compact1by1(self, x):
        """
        Inverse of 'part1by1'. Removes all bits not at indices divisible by 2
        """
        x &= 0x55555555                  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        x = (x | (x >> 1)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x | (x >> 2)) & 0x0f0f0f0f  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x | (x >> 4)) & 0x00ff00ff  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x | (x >> 8)) & 0x0000ffff  # x = ---- ---- ---- ---- fedc ba98 7654 3210
        return x

    def morton_decode(self, code):
        """
        Given an integer value, i.e. 222, it decodes this to i.e. (6, 7, 1) for
        a 3D image or (2, 1, 3, 3) for 4D etc.
        """
        return tuple([self.compact_func(code >> i) for i in range(self.dim)])

    def fill(self, fill_val, dtype=None):
        """Fills the entire internal representation with a given value"""
        self.data = np.full_like(self.data, fill_val, dtype=dtype)

    def to_numpy(self):
        """Transform the data representation to a Numoy array"""
        ret_data = np.zeros(self.shape, dtype=self.data.dtype)
        shape_ranges = (range(val) for val in self.shape)
        for key in product(*shape_ranges):
            idx = self.morton_encode(key)
            ret_data[key] = self.data[idx]
        return ret_data

    def __setitem__(self, key, value):
        """
        Sets the value at the correct place using morton ordering and also
        sends a store operation at this address to the cache
        """
        idx = self.morton_encode(key)
        if self.cache:
            self.cache.store(8*(idx + self.offset), length=8)
        self.data.__setitem__(idx, value)

    def __getitem__(self, key):
        """
        Gets the value at the correct place using morton ordering and also
        sends a load operation at this address to the cache
        """
        idx = self.morton_encode(key)
        if self.cache:
            self.cache.load(8*(idx + self.offset), length=8)
        return self.data.__getitem__(idx)

    def __repr__(self):
        """
        Returns a string representation of the data after it has been
        reshaped to orignal dimensions
        """
        return self.to_numpy().__repr__().replace("array", "Morton").replace("\n ", "\n  ")
