import numpy as np
from itertools import product

from data_structures.caching_data_stucture import CachingDataStructure


# A BlockArray to be used as a caching data structure
class BlockArray(CachingDataStructure):
    def __init__(self, picture=None, shape=None, S=8, cache=None, offset=0):
        assert not (picture is None and shape is None), \
            "You have to set *either* picture or shape"
        assert not (picture is not None and shape is not None), \
            "You can't set *both* picture and shape"
        self.cache = cache
        self.offset = offset
        self.S = S
        if picture is not None:
            picture = np.array(picture)
            self.dim = len(picture.shape)
            self.shape = picture.shape
            self.__set_internal_index_func__()
            self.block_shape = self.__get_block_shape__(self.shape)
            self.pow = int(np.log2(S))
            self.__set_vals__(picture)
        else:
            self.dim = len(shape)
            self.shape = shape
            self.__set_internal_index_func__()
            self.block_shape = self.__get_block_shape__(self.shape)
            self.pow = int(np.log2(S))
            self.data = np.zeros(np.prod(shape))

    def __get_block_shape__(self, shape):
        """
        Sets the shape of the block layout. I.e. if S = 4 and shape = (8, 8),
        the the block_shape will be set to (2, 2)
        """
        return tuple([(0 if shape_i % self.S == 0 else 1) + shape_i // self.S
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

    def get_next_offset(self):
        """
        Returns the offset that the next element should start at if starting
        directly after this element
        """
        return self.offset + 8 * np.prod(self.shape)

    def empty_of_same_shape(self):
        """Returns a BlockArray of same shape as this with all zeros"""
        return BlockArray(shape=self.shape, S=self.S, cache=self.cache, offset=self.get_next_offset())

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
        block_shape_ranges = (range(val) for val in self.block_shape)
        for block_key in product(*block_shape_ranges):
            internal_ranges = (range(self.S) for _ in range(self.dim))
            for idx_key in product(*internal_ranges):
                yield tuple([self.S * block + idx
                             for block, idx in zip(block_key, idx_key)])

    def internal_index(self):
        """
        Returns the block array encoding of coordinates after having been set.
        """
        pass

    def internal_index_2d(self, x, y):
        """
        Given a 2D coordinate, i.e. (6, 7), it returns the
        block array encoding of these
        """
        # Equal to floor division, i.e. x // 16 = x >> log2(16)
        block_x = x >> self.pow
        block_y = y >> self.pow
        block_idx = block_y + block_x * self.block_shape[1]

        # Equal to modulus given S is a power of 2, i.e. x % 16 = x & (16 - 1)
        s_minus_one = self.S - 1
        idx_x = x & s_minus_one
        idx_y = y & s_minus_one

        return self.S * (self.S * block_idx + idx_x) + idx_y

    def internal_index_3d(self, x, y, z):
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
        s_minus_one = self.S - 1
        idx_x = x & s_minus_one
        idx_y = y & s_minus_one
        idx_z = z & s_minus_one
        return self.S * (self.S * (self.S * block_idx + idx_x) + idx_y) + idx_z

    def internal_index_general(self, *args):
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
        s_minus_one = self.S - 1
        idxs = [val & s_minus_one for val in args]

        return block_idx * self.S**self.dim + \
            sum(idxs[i] * self.S**(self.dim-1-i) for i in range(self.dim))

    def fill(self, fill_val, dtype=None):
        """Fills the entire internal representation with a given value"""
        self.data = np.full_like(self.data, fill_val, dtype=dtype)

    def to_numpy(self):
        """Transform the data representation to a Numoy array"""
        ret_data = np.zeros(self.shape, dtype=self.data.dtype)
        shape_ranges = (range(val) for val in self.shape)
        for key in product(*shape_ranges):
            idx = self.internal_index(*key)
            ret_data[key] = self.data[idx]
        return ret_data

    def __setitem__(self, key, value):
        """
        Sets the value at the correct place using block array encoding and also
        sends a store operation at this address to the cache
        """
        idx = self.internal_index(*key)
        if self.cache:
            self.cache.store(8*(idx + self.offset), length=8)
        self.data.__setitem__(idx, value)

    def __getitem__(self, key):
        """
        Gets the value at the correct place using block array encoding and also
        sends a load operation at this address to the cache
        """
        idx = self.internal_index(*key)
        if self.cache:
            self.cache.load(8*(idx + self.offset), length=8)
        return self.data.__getitem__(idx)

    def __repr__(self):
        """
        Returns the Numpy representation of the data after it has been
        reshaped to orignal dimensions
        """
        return self.to_numpy().__repr__().replace("array", "Block").replace("\n ", "\n  ")
