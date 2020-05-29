from cachesim import CacheSimulator, Cache, MainMemory
import matplotlib.pyplot as plt
import numpy as np

MORTON, ROW_ARR, BLOCK_ARR = "morton", "row_arr", "block_arr"
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
        'MISS': 1.0  # Miss will never occur for MEM
    },
}


def save_fig(filename: str, path: str = "./",
             should_save: bool = True) -> None:
    """
    Saves the figure in a tight format with the given filename at path
    if dictated by should_save
    """
    if should_save:
        new_filename = path + filename.replace("_", "-")
        # Add .pdf to file ending if not already part of it
        filename_splitted = new_filename.split(".")
        if len(filename_splitted) == 1 or \
           filename_splitted[-1] not in ["pdf", "png"]:
            new_filename += ".pdf"
        plt.savefig(new_filename.lower(), bbox_inches='tight')


def make_cache() -> CacheSimulator:
    """Returns a fresh cache of standard size"""
    mem = MainMemory()
    l3 = Cache("L3", 20480, 16, 64, "LRU")  # 20MB: 20480 sets, 16-ways with
                                            # cacheline size of 64 bytes
    mem.load_to(l3)
    mem.store_from(l3)
    l2 = Cache("L2", 512, 8, 64, "LRU", store_to=l3, load_from=l3)  # 256KB
    l1 = Cache("L1", 64, 8, 64, "LRU", store_to=l2, load_from=l2)  # 32KB
    cs = CacheSimulator(l1, mem)
    return cs


def get_val_arr(results: dict, data_typ: str,
                cache_level: int = 0, stat: str = "HIT_count") -> np.ndarray:
    """
    Returns an ndarray of the corresponding y-values for all the n-values for
    a given data_typ, cache_level and stat
    """
    sorted_res = sorted(results.items(), key=lambda key_val: int(key_val[0]))
    return np.array([float(val[data_typ][cache_level][stat])
                     for _, val in sorted_res])


def get_val_arr_time(results: dict, data_typ: str) -> list:
    """
    Returns an array of the simulated cache time for all n-values for a
    specific data_typ based on the number of hits and misses at different
    cache levels
    """
    res_arr = []
    for n in results:
        time_taken = 0.0
        for cache_level in range(len(results[n][data_typ])):
            for typ in ['HIT', 'MISS']:
                count = results[n][data_typ][cache_level][f"{typ}_count"]
                time_taken += count * SIM_CACHE_TIMES[cache_level][typ]
        res_arr.append(time_taken)
    return res_arr


def prettify_name(data_typ: str) -> str:
    """
    Returns a pretty version of the data_typ name. If none is found, the
    data_typ is simply returned
    """
    pretty_names = {
        MORTON: "Morton Ordering",
        ROW_ARR: "Row Major Array",
        BLOCK_ARR: "Block Array"
    }
    return pretty_names[data_typ] if data_typ in pretty_names else data_typ


def num_column_and_width_to_displacement(n: int, width: float = None) \
                                         -> tuple:
    if width is None:
        width = (1.0 / float(n)) * 0.7
    div = n // 2
    arr = [i+0.5 for i in range(-div, div)] \
        if n % 2 == 0 \
        else [i for i in range(-div, div+1)]
    return np.array(arr) * width, width
