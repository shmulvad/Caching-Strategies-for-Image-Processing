from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Generator


class CachingDataStructure(ABC):
    shape: tuple
    dim: int
    offset: int
    name: str
    cache = None
    data = None

    def __init__(self):
        """
        A generic CachingDataStructure. When initializing, either a picture
        as an array or a shape as a tuple should be supplied. The supplied
        shape/picture should be a perfect square/cube/etc. If given a cache
        of pycachesim type, it will also simulato loading and storing to this
        cache.
        """
        super().__init__()

    def __setitem__(self, key: tuple, value: Any) -> None:
        """
        Sets the value at the correct place using row major array encoding and
        also sends a store operation at this address to the cache
        """
        idx = self.internal_index(*key)
        if self.cache:
            self.cache.store(8*(idx + self.offset), length=8)
        self.data.__setitem__(idx, value)

    def __getitem__(self, key: tuple) -> Any:
        """
        Gets the value at the correct place using row major array encoding and
        also sends a load operation at this address to the cache
        """
        idx = self.internal_index(*key)
        if self.cache:
            self.cache.load(8*(idx + self.offset), length=8)
        return self.data.__getitem__(idx)

    def __repr__(self) -> str:
        """
        Returns the Numpy representation of the data after it has been
        reshaped to orignal dimensions
        """
        padding = " " * (len(self.name) - len("array"))
        return self.to_numpy().__repr__().replace("array", self.name).\
            replace("\n ", f"\n {padding}")

    def fill(self, fill_val: Any, dtype=None) -> 'CachingDataStructure':
        """Fills the entire internal representation with a given value"""
        self.data = np.full_like(self.data, fill_val, dtype=dtype)
        return self

    def valid_index(self, *args: int, pad: int = 0) -> bool:
        """
        Returns true the index specified is valid in this data structure and
        within optional padding value
        """
        return len(args) == self.dim and \
            all([args[i] >= pad and args[i] < self.shape[i] - pad
                 for i in range(self.dim)])

    def get_next_offset(self) -> int:
        """
        Returns the offset that the next CachingDataStructure should start at
        if starting directly after this CachingDataStructure
        """
        return self.offset + 8 * np.prod(self.shape)

    @abstractmethod
    def empty_of_same_shape(self) -> 'CachingDataStructure':
        """
        Returns a new CachingDataStructure of same type and shape with all
        0-values
        """
        pass

    @abstractmethod
    def internal_index(self, *args: int) -> int:
        """
        Computes the internal 1D index at a given index in CachingDataStructure
        I.e. we could have internal_index(0, 0, 0) -> 0
        """
        pass

    @abstractmethod
    def iter_keys(self) -> Generator[tuple, None, None]:
        """
        Returns a generator that yields tuples of the keys in
        internal linear layout (optimal spatial locality)
        """
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Transform the CachingDataStructure to a Numpy array"""
        pass
