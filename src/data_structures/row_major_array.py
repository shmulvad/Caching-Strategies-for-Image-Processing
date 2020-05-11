import numpy as np
from itertools import product

from data_structures.caching_data_stucture import CachingDataStructure


# A decorator for the NdArray that implements ICachingDataStructure to allow it
# to be used in the same functions
class RowMajorArray(CachingDataStructure):
    def __init__(self, picture=None, shape=None, cache=None, offset=0):
        assert not (
            picture is None and shape is None), "You have to set *either* picture or shape"
        assert not (
            picture is not None and shape is not None), "You can't set *both* picture and shape"
        self.cache = cache
        self.offset = offset
        if picture is not None:
            picture = np.array(picture)
            self.dim = len(picture.shape)
            self.shape = picture.shape
            self.__set_internal_index_func__()
            self.__set_vals__(picture)
        else:
            self.dim = len(shape)
            self.shape = shape
            self.__set_internal_index_func__()
            self.data = np.zeros(np.prod(self.shape))

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

    def __set_vals__(self, picture):
        """Sets the internal data object to values of image."""
        data = np.zeros(np.prod(self.shape), dtype=picture.dtype)
        shape_ranges = (range(val) for val in self.shape)
        for key in product(*shape_ranges):
            data[self.internal_index(*key)] = picture[key]
        self.data = data

    def get_next_offset(self):
        """
        Returns the offset that the next element should start at if starting
        directly after this element
        """
        return self.offset + 8 * np.prod(self.shape)

    def empty_of_same_shape(self):
        """Returns a RowMajorArray of same shape as this with all zeros"""
        return RowMajorArray(shape=self.shape, cache=self.cache,
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
        shape_ranges = (range(val) for val in self.shape)
        for key in product(*shape_ranges):
            yield key

    def internal_index(self):
        """
        Returns the block array encoding of coordinates after having been set.
        """
        pass

    def internal_index_2d(self, x, y):
        """
        Given a 2D coordinate, i.e. (6, 7), it returns the
        row major array encoding of these
        """
        return self.shape[1] * x + y

    def internal_index_3d(self, x, y, z):
        """
        Given a 3D coordinate, i.e. (6, 7, 1), it returns the
        row major array encoding of these
        """
        return self.shape[2] * (self.shape[1] * x + y) + z

    def internal_index_general(self, *args):
        """
        Given an arbitrary number of coordinates, i.e. (6, 7, 1, 4, 1), it
        returns the row major array encoding of these (as long as number of
        arguments are equal to dim)
        """
        assert len(args) == self.dim, \
            f"Number of args ({len(args)}) does not match up with \
                internal dimension ({self.dim})."
        # The following is the general case of something like
        # x * self.shape[1] * self.shape[2] + y * self.shape[2] + z
        return sum(args[i] * int(np.prod(self.shape[i+1:]))
                   for i in range(self.dim))

    def fill(self, fill_val, dtype=None):
        """Fills the entire internal representation with a given value"""
        self.data = np.full_like(self.data, fill_val, dtype=dtype)
        return self

    def to_numpy(self):
        """Transform the data representation to a Numoy array"""
        return self.data.reshape(self.shape)

    def __setitem__(self, key, value):
        """
        Sets the value at the correct place using row major array encoding and
        also sends a store operation at this address to the cache
        """
        idx = self.internal_index(*key)
        if self.cache:
            self.cache.store(8*(idx + self.offset), length=8)
        self.data.__setitem__(idx, value)

    def __getitem__(self, key):
        """
        Gets the value at the correct place using row major array encoding and
        also sends a load operation at this address to the cache
        """
        idx = self.internal_index(*key)
        if self.cache:
            self.cache.load(8*(idx + self.offset), length=8)
        return self.data.__getitem__(idx)

    def __repr__(self):
        """
        Returns a string representation of the data after it has been
        reshaped to orignal dimensions
        """
        return self.to_numpy().__repr__().replace("array", "rm_arr").replace("\n ", "\n  ")
