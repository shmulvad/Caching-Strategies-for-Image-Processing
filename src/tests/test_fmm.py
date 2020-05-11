import numpy as np
import pytest

from data_structures.row_major_array import RowMajorArray
from algorithms.fmm import fmm


class TestFastMarchingMethod(object):
    def setup_method(self, f):
        np.random.seed(0)

    def test_something(self):
        pass
