from abc import ABC, abstractmethod


class CachingDataStructure(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def valid_index(self, *args, pad=0):
        pass

    @abstractmethod
    def empty_of_same_shape(self):
        pass

    @abstractmethod
    def internal_index(self, *args):
        pass

    @abstractmethod
    def iter_keys(self):
        pass

    @abstractmethod
    def get_next_offset(self):
        pass
