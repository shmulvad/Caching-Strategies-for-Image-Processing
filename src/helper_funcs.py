from cachesim import CacheSimulator, Cache, MainMemory
import matplotlib.pyplot as plt
import numpy as np

MORTON, ROW_ARR, BLOCK_ARR = "morton", "row_arr", "block_arr"


def save_fig(filename: str, path: str = "./",
             should_save: bool = True) -> None:
    """
    Saves the figure in a tight format with the given filename at path
    if dictated by should_save.
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
    """Returns a fresh cache of standard size."""
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
    a given data_typ, cache_level and stat.
    """
    sorted_res = sorted(results.items(), key=lambda key_val: int(key_val[0]))
    return np.array([float(val[data_typ][cache_level][stat])
                     for _, val in sorted_res])


def prettify_name(data_typ: str) -> str:
    """
    Returns a pretty version of the data_typ name. If none is found, the
    data_typ is simply returned.
    """
    pretty_names = {
        MORTON: "Morton Ordering",
        ROW_ARR: "Row Major Array",
        BLOCK_ARR: "Block Array"
    }
    return pretty_names[data_typ] if data_typ in pretty_names else data_typ


def print_table_nums(dim: int, results: dict, ns: list, stat: str) -> None:
    """
    Prints a stat (i.e. HIT_count) for morton ordering and block array
    relative to row-major array. Is printed in a format that is copy-pasteable
    on a row basis to a LaTeX-table.
    """
    print(f"Dim: {dim}\n{'ns':10}: {ns}")
    row_arr_res = get_val_arr(results, ROW_ARR, stat=stat)
    for data_typ in [MORTON, BLOCK_ARR]:
        rel_res = get_val_arr(results, data_typ, stat=stat) / row_arr_res
        rel_res_strs = map(lambda val: f"{val:.2f}", rel_res)
        print(f"{data_typ:10}: {' & '.join(rel_res_strs)}")
    print("\n")


def hit_to_misses(dim: int, results: dict, ns: list) -> None:
    """
    Prints the hit-to-miss ratio for the three data structures. Is printed in
    a format that is copy-pasteable on a row basis to a LaTeX-table.
    """
    print(f"Dim: {dim}\n{'ns':10}: {ns}")
    for data_typ in [MORTON, ROW_ARR, BLOCK_ARR]:
        hits = get_val_arr(results, data_typ, stat="HIT_count")
        misses = get_val_arr(results, data_typ, stat="MISS_count")
        hit_to_misses = map(lambda val: f"{val:.2f}", hits / misses)
        print(f"{data_typ:10}: {' & '.join(hit_to_misses)}")
    print("\n")
