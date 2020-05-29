import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import concurrent.futures
from itertools import product, repeat
import numpy as np
import json

from helper_funcs import save_fig, prettify_name, MORTON, ROW_ARR, BLOCK_ARR
from updater_class import UpdateProgress

from data_structures.caching_data_stucture import CachingDataStructure
from data_structures.morton_order import MortonOrder
from data_structures.block_array import BlockArray
from data_structures.row_major_array import RowMajorArray

# Set to true if you want to run the tests again.
# Otherwise just loads results from JSON
GENERATE_NEW_RESULTS = True

# Set to true if you want to save figures to disk. Change path as needed
SAVE_FIGURES_TO_DISK = True
FIG_SAVE_PATH = "../../thesis/figures/props/"

# Global constants for test. Change these if you want to run another test
MAX_PROP_LEVEL = 32
SHAPE = (128, 128, 128)
MID = SHAPE[0] // 2
START_POINTS = [(0, 0, 0), (MID, MID, MID)]
DATA_ARRS = [
    MortonOrder(shape=SHAPE),
    RowMajorArray(shape=SHAPE),
    BlockArray(shape=SHAPE)
]

# Global constants used for type-safety when accessing properties in
# dictionary. Do not change
INF = float("inf")
XS, YS1, YS2 = "xs", "ys1", "ys2"
META, MAX_PROP_LEVEL_STR, SHAPE_STR = "meta", "max_prop_level", "shape"

sns.set(font_scale=1.9)
matplotlib.rcParams['figure.figsize'] = (1.1*18.0, 1.1*4.8)


def indices_for_prop_level_2d(prop_level: int, start_point: tuple) -> set:
    """
    Returns the indices at a given propagation level away
    from start_point in 2D
    """
    x, y = start_point
    indices = set([])
    # Will double add corners, but no worries because we use a set
    for i in range(-prop_level, prop_level + 1):
        indices.add((x - prop_level, y + i))
        indices.add((x + prop_level, y + i))

        indices.add((x + i, y - prop_level))
        indices.add((x + i, y + prop_level))
    return indices


def indices_for_prop_level_3d(prop_level: int, start_point: tuple) -> set:
    """
    Returns the indices at a given propagation level away from start_point
    in 3D
    """
    x, y, z = start_point
    indices = set([])
    # Will double add corners, but no worries because we use a set
    for i in range(-prop_level, prop_level + 1):
        for j in range(-prop_level, prop_level + 1):
            indices.add((x - prop_level, y + i, z + j))
            indices.add((x + prop_level, y + i, z + j))

            indices.add((x + i, y - prop_level, z + j))
            indices.add((x + i, y + prop_level, z + j))

            indices.add((x + i, y + j, z - prop_level))
            indices.add((x + i, y + j, z + prop_level))
    return indices


def propagation_level_end(start_point: tuple, shape: tuple) -> int:
    """
    Returns the maximum propagation level where our data points will change
    I.e. if start_point=(8, 8, 8) and shape=(16, 16, 16), then returns 8
    If start_point=(1, 1, 3) and shape=(16, 16, 16) then returns 16-1=15
    """
    max_val = -1
    for start_val, shape_val in zip(start_point, shape):
        abs_val = abs(shape_val - start_val)
        max_val = max_val if max_val > abs_val else abs_val
    return max_val


def total_dist_from_point(data: CachingDataStructure, start_point: tuple,
                          max_prop_level: int = INF) -> tuple:
    """
    For some CachingDataType, it calculates the distance in linear
    space for all different propagation levels. Returns a tuple of three
    arrays; (1) an array of propagation level, (2) summed linear distance and
    (3) summed linear distance divided by the number of pixels/voxels at that
    propagation level
    """
    assert data.dim == len(start_point), \
        "Dimensions of data and start point don't match"
    assert data.dim in [2, 3], "Data should be 2- or 3-dimensional"

    start_internal_index = data.internal_index(*start_point)
    get_indices = indices_for_prop_level_2d \
        if data.dim == 2 \
        else indices_for_prop_level_3d

    prop_level_end = propagation_level_end(start_point, data.shape)
    prop_levels = range(min(prop_level_end, max_prop_level))
    ys1, ys2 = [], []
    summ, num_of_voxels = 0, 0
    for prop_level in prop_levels:
        for key in get_indices(prop_level, start_point):
            if data.valid_index(*key):
                summ += abs(data.internal_index(*key) - start_internal_index)
                num_of_voxels += 1
        ys1.append(summ)
        ys2.append(summ / num_of_voxels)
    return np.array(prop_levels), np.array(ys1), np.array(ys2)


def get_avg_of_prop(data: CachingDataStructure, max_prop_level: int) -> tuple:
    """
    Gets the average linear distance at a given propagation level for a
    CachingDataStructure when propagating to a certain max_prop_level
    """
    n, dim = data.shape[0], data.dim
    assert max_prop_level < n // 2, \
        "Max prop level should be lower than half of side length"
    # Generate all possible permutations of points in inner cube
    ranges = tuple(range(max_prop_level, n - max_prop_level)
                   for _ in range(dim))
    points = product(*ranges)

    xs, ys1, ys2 = (np.zeros(max_prop_level) for _ in range(3))
    total_points = (n - 2 * max_prop_level)**dim
    updater = UpdateProgress(total_points)
    updater.update_status(0, data.print_name)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, (xss, yss1, yss2) in enumerate(executor.map(
          total_dist_from_point,
          repeat(data), points, repeat(max_prop_level))):
            xs = xss
            ys1 += yss1
            ys2 += yss2

            if i % 10 == 0:
                updater.update_status(i, data.print_name)
    return xs, ys1 / max_prop_level, ys2 / max_prop_level


def total_dist_from_point_helper(data: CachingDataStructure,
                                 start_point: tuple,
                                 max_prop_level: int = INF) -> tuple:
    """
    Calls total_dist_from_point but also returns print_name and start_point,
    making it easier to use in ProcessPoolExecutor
    """
    xs, ys1, ys2 = total_dist_from_point(data, start_point, max_prop_level)
    return data.print_name, start_point, xs, ys1, ys2


def gen_prop_dist_res(data_arrs: list, start_points: tuple = None) -> dict:
    """
    Generates the propagation data for an array of CachingDataStructures at a
    number of start points
    """
    results_prop = {
        META: {
            SHAPE_STR: data_arrs[0].shape
        }
    }
    data_arrs_iter, start_points_iter = zip(*product(data_arrs, start_points))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for (name, point, xs, ys1, ys2) in executor.map(
          total_dist_from_point_helper, data_arrs_iter, start_points_iter):
            if name not in results_prop:
                results_prop[name] = {}
            results_prop[name][str(point)] = {
                XS: xs.tolist(),
                YS1: ys1.tolist(),
                YS2: ys2.tolist()
            }
    return results_prop


def gen_avg_prop_dist_res(data_arrs: list, max_prop_level: int) -> dict:
    """
    Generates the average of the propagation data for an array of
    CachingDataStructures
    """
    results_avg_prop = {
        META: {
            MAX_PROP_LEVEL_STR: MAX_PROP_LEVEL,
            SHAPE_STR: SHAPE
        }
    }
    for data in data_arrs:
        xs, ys1, ys2 = get_avg_of_prop(data, max_prop_level)
        results_avg_prop[data.print_name] = {
            XS: xs.tolist(),
            YS1: ys1.tolist(),
            YS2: ys2.tolist()
        }
    return results_avg_prop


def plot_prop_dist(results: dict, start_point: tuple,
                   fig_save_path: str = "./",
                   save_figs: bool = True) -> None:
    """
    Plots the results for propagation for all CachingDataStructure types at a
    given start point. If no start_point is none, the average will be
    plotted instead
    """
    use_avg = start_point is None
    label_start = "Average\nsummed" if use_avg else "Summed"
    ys1_label = f"{label_start} linear distance"
    ys2_label = f"{label_start} linear distance\ndivided by number of voxels"

    fig, axes = plt.subplots(2, 3, figsize=(17, 10),
                             sharex="col", sharey="row",
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    for axes_row, use_ys1 in zip(axes, [True, False]):
        for ax, data_typ in zip(axes_row, [MORTON, ROW_ARR, BLOCK_ARR]):
            data = results[data_typ] if use_avg \
                else results[data_typ][str(start_point)]

            xs, ys = data[XS], data[YS1 if use_ys1 else YS2]
            ax.plot(xs, ys)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set(xlabel="Propagation level",
                   ylabel=(ys1_label if use_ys1 else ys2_label))
            if use_ys1:
                ax.set_title(prettify_name(data_typ))
            ax.label_outer()

    filename = f"prop-dist-{'avg' if use_avg else start_point[0]}.pdf"
    save_fig(filename, fig_save_path, save_figs)
    plt.show()


if __name__ == "__main__":
    if GENERATE_NEW_RESULTS:
        results_prop = gen_prop_dist_res(DATA_ARRS, START_POINTS)
        with open('results/prop-dist.json', 'w') as f:
            json.dump(results_prop, f, indent=4)

        results_avg_prop = gen_avg_prop_dist_res(DATA_ARRS, MAX_PROP_LEVEL)
        with open('results/prop-dist-avg.json', 'w') as f:
            json.dump(results_avg_prop, f, indent=4)

    with open('results/prop-dist.json', 'r') as f:
        results_prop = json.load(f)
    with open('results/prop-dist-avg.json', 'r') as f:
        results_avg_prop = json.load(f)

    # Plotting
    for start_point in START_POINTS:
        plot_prop_dist(results_prop, start_point,
                       FIG_SAVE_PATH, SAVE_FIGURES_TO_DISK)
    plot_prop_dist(results_avg_prop, None, FIG_SAVE_PATH, SAVE_FIGURES_TO_DISK)
