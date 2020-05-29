import numpy as np
from itertools import product
from typing import Generator
from cachesim import CacheSimulator

from helper_funcs import BLOCK_ARR
from data_structures.caching_data_stucture import CachingDataStructure


# A BlockArray to be used as a caching data structure
class BlockArray(CachingDataStructure):
    print_name = BLOCK_ARR
    type_name = "BlockArray"

    def __init__(self, picture: np.ndarray = None, shape: tuple = None,
                 cache: CacheSimulator = None, offset: int = 0, K: int = 8):
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
            self.__set_K__(K)
            self.__set_internal_index_func__()
            self.block_shape = self.__get_block_shape__(self.shape)
            self.pow = int(np.log2(self.K))
            self.__set_vals__(picture)
        else:
            self.dim = len(shape)
            self.__set_shape__(shape)
            self.__set_K__(K)
            self.__set_internal_index_func__()
            self.block_shape = self.__get_block_shape__(self.shape)
            self.pow = int(np.log2(self.K))
            self.data = np.zeros(np.prod(shape))

    def __set_K__(self, K: int):
        """
        Sets the value of the K-variable, making sure it is not higher than
        the of a single dimension
        """
        self.K = self.shape[0] if self.shape[0] in [2, 4] else K

    def __get_block_shape__(self, shape: tuple) -> tuple:
        """
        Sets the shape of the block layout. I.e. if S = 4 and shape = (8, 8),
        the the block_shape will be set to (2, 2)
        """
        return tuple([(0 if shape_i % self.K == 0 else 1) + shape_i // self.K
                      for shape_i in shape])

    def __set_vals__(self, picture):
        """Sets the internal data object to values of image."""
        data = np.zeros(np.prod(self.shape), dtype=picture.dtype)
        shape_ranges = (range(val) for val in self.shape)
        for key in product(*shape_ranges):
            data[self.internal_index(*key)] = picture[key]
        self.data = data

    def __set_internal_index_func__(self):
        """
        Sets the function to compute internal index based on the dimensionality
        If 2- or 3D, then specific functions for these have been made. If
        another dimension, use general but also slower function
        """
        if self.dim == 2:
            self.internal_index = self.internal_index_2d
        elif self.dim == 3:
            self.internal_index = self.internal_index_3d
        else:
            self.internal_index = self.internal_index_general

    def get_next_offset(self) -> int:
        """
        Returns the offset that the next element should start at if starting
        directly after this element
        """
        return self.offset + 8 * np.prod(self.shape)

    def empty_of_same_shape(self) -> 'BlockArray':
        """Returns a BlockArray of same shape as this with all zeros"""
        return BlockArray(shape=self.shape, K=self.K, cache=self.cache,
                          offset=self.get_next_offset())

    def iter_keys(self) -> Generator[tuple, None, None]:
        """
        Returns a generator that yields tuples of the keys in
        internal linear layout (optimal spatial locality)
        """
        block_shape_ranges = (range(val) for val in self.block_shape)
        for block_key in product(*block_shape_ranges):
            internal_ranges = (range(self.K) for _ in range(self.dim))
            for idx_key in product(*internal_ranges):
                yield tuple([self.K * block + idx
                             for block, idx in zip(block_key, idx_key)])

    def internal_index(self):
        """
        Returns the block array encoding of coordinates after having been set.
        """
        pass

    def internal_index_2d(self, x: int, y: int) -> int:
        """
        Given a 2D coordinate, i.e. (6, 7), it returns the
        block array encoding of these
        """
        # Equal to floor division, i.e. x // 16 = x >> log2(16)
        block_x = x >> self.pow
        block_y = y >> self.pow
        block_idx = block_y + block_x * self.block_shape[1]

        # Equal to modulus given S is a power of 2, i.e. x % 16 = x & (16 - 1)
        s_minus_one = self.K - 1
        idx_x = x & s_minus_one
        idx_y = y & s_minus_one

        return self.K * (self.K * block_idx + idx_x) + idx_y

    def internal_index_3d(self, x: int, y: int, z: int) -> int:
        """
        Given a 3D coordinate, i.e. (6, 7, 1), it returns the
        block array encoding of these
        """
        # Equal to floor division, i.e. x // 16 = x >> log2(16)
        block_x = x >> self.pow
        block_y = y >> self.pow
        block_z = z >> self.pow
        block_idx = self.block_shape[2] * \
            (self.block_shape[1] * block_x + block_y) + block_z

        # Equal to modulus given S is a power of 2, i.e. x % 16 = x & (16 - 1)
        s_minus_one = self.K - 1
        idx_x = x & s_minus_one
        idx_y = y & s_minus_one
        idx_z = z & s_minus_one
        return self.K * (self.K * (self.K * block_idx + idx_x) + idx_y) + idx_z

    def internal_index_general(self, *args: int) -> int:
        """
        Given an arbitrary number of coordinates, i.e. (6, 7, 1, 1, 5), it
        returns the block array encoding of these (as long as number of
        arguments are equal to dim)
        """
        assert len(args) == self.dim, \
            f"Number of args ({len(args)}) does not match up with \
                internal dimension ({self.dim})."
        # Equal to floor division, i.e. x // 16 = x >> log2(16)
        block = [val >> self.pow for val in args]
        block_idx = sum(block[i] * int(np.prod(self.block_shape[i+1:]))
                        for i in range(self.dim))

        # Equal to modulus given S is a power of 2, i.e. x % 16 = x & (16 - 1)
        s_minus_one = self.K - 1
        idxs = [val & s_minus_one for val in args]

        return block_idx * self.K**self.dim + \
            sum(idxs[i] * self.K**(self.dim-1-i) for i in range(self.dim))
