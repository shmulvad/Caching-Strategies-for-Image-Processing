import numpy as np
from itertools import product
from typing import Generator

from data_structures.caching_data_stucture import CachingDataStructure


# A decorator for the NdArray that implements ICachingDataStructure to allow it
# to be used in the same functions
class RowMajorArray(CachingDataStructure):
    print_name = "rm_arr"
    type_name = "RowMajorArray"

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
            self.__set_shape__(picture.shape)
            self.__set_internal_index_func__()
            self.__set_vals__(picture)
        else:
            self.dim = len(shape)
            self.__set_shape__(shape)
            self.__set_internal_index_func__()
            self.data = np.zeros(np.prod(self.shape))

    def __set_internal_index_func__(self) -> None:
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

    def __set_vals__(self, picture) -> None:
        """Sets the internal data object to values of image."""
        data = np.zeros(np.prod(self.shape), dtype=picture.dtype)
        shape_ranges = (range(val) for val in self.shape)
        for key in product(*shape_ranges):
            data[self.internal_index(*key)] = picture[key]
        self.data = data

    def empty_of_same_shape(self) -> 'RowMajorArray':
        """Returns a RowMajorArray of same shape as this with all zeros"""
        return RowMajorArray(shape=self.shape, cache=self.cache,
                             offset=self.get_next_offset())

    def iter_keys(self) -> Generator[tuple, None, None]:
        """
        Returns a generator that yields a tuple of the keys in
        internal linear layout (optimal spatial locality)
        """
        shape_ranges = (range(N) for N in self.shape)
        for key in product(*shape_ranges):
            yield key

    def internal_index(self):
        """
        Returns the block array encoding of coordinates after having been set.
        """
        pass

    def internal_index_2d(self, x: int, y: int) -> int:
        """
        Given a 2D coordinate, i.e. (6, 7), it returns the
        row major array encoding of these
        """
        return self.shape[1] * x + y

    def internal_index_3d(self, x: int, y: int, z: int) -> int:
        """
        Given a 3D coordinate, i.e. (6, 7, 1), it returns the
        row major array encoding of these
        """
        return self.shape[2] * (self.shape[1] * x + y) + z

    def internal_index_general(self, *args: int) -> int:
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
