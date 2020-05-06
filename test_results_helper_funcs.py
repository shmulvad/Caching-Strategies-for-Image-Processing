from cachesim import CacheSimulator, Cache, MainMemory
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


MORTON = "morton"
ROW_ARR = "row_arr"
BLOCK_ARR = "block_arr"
SIM_CACHE_TIMES = {
    0: {
        'HIT':  0.000001,
        'MISS': 0.0001
    },
    1: {
        'HIT':  0.0001,
        'MISS': 0.01
    },
    2: {
        'HIT':  0.01,
        'MISS': 1.0
    },
    3: {
        'HIT':  1.5,
        'MISS': 1.0 # Miss will never occur for MEM
    },
}


def save_fig(filename, path="./", should_save=True):
    if should_save:
        new_filename = path + filename.replace("_", "-")
        plt.savefig(new_filename, bbox_inches = 'tight')


def make_cache():
    mem = MainMemory()
    l3 = Cache("L3", 20480, 16, 64, "LRU")  # 20MB: 20480 sets, 16-ways with cacheline size of 64 bytes
    mem.load_to(l3)
    mem.store_from(l3)
    l2 = Cache("L2", 512, 8, 64, "LRU", store_to=l3, load_from=l3)  # 256KB
    l1 = Cache("L1", 64, 8, 64, "LRU", store_to=l2, load_from=l2)  # 32KB
    cs = CacheSimulator(l1, mem)
    return cs


def get_val_arr(results, data_typ, cache_level=0, stat="HIT_count"):
    sorted_res = sorted(results.items(), key=lambda key_val: int(key_val[0]))
    return np.array([float(val[data_typ][cache_level][stat])
                     for _, val in sorted_res])


def get_val_arr_time(results, data_typ):
    res_arr = []
    for n in results:
        time_taken = 0.0
        for cache_level in range(len(results[n][data_typ])):
            for typ in ['HIT', 'MISS']:
                count = results[n][data_typ][cache_level][f"{typ}_count"]
                time_taken += count * SIM_CACHE_TIMES[cache_level][typ]
        res_arr.append(time_taken)
    return res_arr


def num_column_and_width_to_displacement(n, width):
    div = n // 2
    arr = [i+0.5 for i in range(-div, div)] \
          if n % 2 == 0 \
          else [i for i in range(-div, div+1)]
    return np.array(arr) * width
