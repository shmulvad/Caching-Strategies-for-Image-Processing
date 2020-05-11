import numpy as np

from data_structures.morton_order import MortonOrder
from data_structures.block_array import BlockArray
from data_structures.row_major_array import RowMajorArray

from helper_funcs import make_cache


class TestCache(object):
    def setup_method(self, f):
        pass

    def test_same_internal_acces_pattern_yields_same_stats(self):
        vals = np.random.rand(16, 16, 16)
        morton = MortonOrder(vals, cache=make_cache())
        block_arr = BlockArray(vals, cache=make_cache())
        row_arr = RowMajorArray(vals, cache=make_cache())

        stats = []
        for data in [morton, block_arr, row_arr]:
            for is_warm_up in [True, False]:
                res = data.empty_of_same_shape()
                for key in data.iter_keys():
                    res[key] = 2 * data[key]
                data.cache.force_write_back()
                if not is_warm_up:
                    stats.append(list(data.cache.stats()))
                data.cache.reset_stats()
        assert stats[0] == stats[1] and stats[1] == stats[2]
