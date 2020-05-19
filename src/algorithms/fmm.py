from heapq import heapify, heappush, heappop
import numpy as np
import math
from data_structures.caching_data_stucture import CachingDataStructure

INF = float("inf")
EPS = 0.05
KNOWN = True


class Neighbor(object):
    """
    A Neighbour to be used in min-heap.
    Defines the time value, coordinates and if it is valid
    """
    def __init__(self, min_val: float, coords: list, valid: bool):
        self.min_val = min_val
        self.coords = coords
        self.valid = valid

    def __repr__(self) -> str:
        """Returns a string representation of the Neighbor"""
        return f"Neighbor({self.min_val:.3f}, {self.coords}, {self.valid})"

    def __lt__(self, other) -> bool:
        """
        Define the less than function between Neighbors.
        Required to be able to use in min-heap
        """
        return self.min_val < other.min_val


###########################
#            2D           #
###########################
def safe_get_times_2d(times: CachingDataStructure, i: int, j: int) -> float:
    """
    Returns the saved time value if is a valid index, otherwise infinity
    """
    return INF if not times.valid_index(i, j) else times[i, j]


def calc_time_2d(speed_func_arr: CachingDataStructure,
                 times: CachingDataStructure,
                 i: int, j: int) -> float:
    """
    Calculates the time value for a newly added neighbor in 2D.
    Implementation based on
    https://en.wikipedia.org/wiki/Eikonal_equation#Numerical_approximation
    """
    u_i = min(
        safe_get_times_2d(times, i - 1, j),
        safe_get_times_2d(times, i + 1, j)
    )
    u_j = min(
        safe_get_times_2d(times, i, j - 1),
        safe_get_times_2d(times, i, j + 1)
    )
    u_sum = u_i + u_j
    diff = abs(u_i - u_j)

    speed_val = speed_func_arr[i, j] + EPS
    reciproc_speed_func_elm = 1.0 / speed_val
    reciproc_speed_func_elm_2 = reciproc_speed_func_elm ** 2

    return 0.5 * (u_sum + np.sqrt(u_sum*u_sum - 2.0 *
                  (u_i * u_i + u_j * u_j - reciproc_speed_func_elm_2))) \
        if diff <= reciproc_speed_func_elm \
        else reciproc_speed_func_elm + min(u_i, u_j)


def get_neighbors_2d(status: CachingDataStructure, i: int, j: int) -> list:
    """
    Returns a list of tuples of all coordinates that are direct neighbors,
    meaning the index is valid and they are not KNOWN
    """
    coords = []
    for (x, y) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
        if status.valid_index(x, y) and not status[x, y]:  # Not known
            coords.append((x, y))
    return coords


def fmm_2d(speed_func_arr: CachingDataStructure,
           start_point: tuple,
           end_point: tuple = None) -> CachingDataStructure:
    """
    Computes the FMM for a 2D speed function and a given start point. If
    an end point is supplied, only the necessary coordinates will be computed.
    Otherwise all coordinates
    """
    i, j = start_point

    times = speed_func_arr.empty_of_same_shape()
    times.fill(INF)
    times[i, j] = 0.0

    # Status[i, j] is false if (i, j) is far or neighbour, true if (i, i) is
    # known
    status = times.empty_of_same_shape()
    status.fill(False, dtype=bool)
    status[i, j] = KNOWN

    heap_pointers = status.empty_of_same_shape()
    heap_pointers.fill(Neighbor(0, 0, False), dtype='object')

    neighbors = []
    heapify(neighbors)

    first_run = True
    while (i, j) != end_point and (first_run or neighbors):
        first_run = False

        # Get new neighbors
        coords = get_neighbors_2d(status, i, j)

        # Calculate arrival times of newly added neighbors
        for (x, y) in coords:
            val_time = calc_time_2d(speed_func_arr, times, x, y)
            if val_time < times[x, y]:
                if heap_pointers[x, y].valid:
                    heap_pointers[x, y].valid = False
                times[x, y] = val_time
                heap_elm = Neighbor(val_time, [x, y], True)
                heap_pointers[x, y] = heap_elm
                heappush(neighbors, heap_elm)

        # Find the smallet value of neighbours
        while neighbors:
            element = heappop(neighbors)
            if element.valid:
                break
        i, j = element.coords

        # Add coordinate of smallest value to known
        status[i, j] = KNOWN
    return times


###########################
#            3D           #
###########################
def safe_get_times_3d(times: CachingDataStructure,
                      i: int, j: int, k: int) -> float:
    """
    Returns the saved time value in 3D array if is a valid index,
    otherwise infinity
    """
    return INF if not times.valid_index(i, j, k) else times[i, j, k]


def calc_time_3d(speed_func_arr: CachingDataStructure,
                 times: CachingDataStructure,
                 i: int, j: int, k: int) -> float:
    """
    Calculates the time value for a newly added neighbor in 3D.
    Implementation based on
    https://en.wikipedia.org/wiki/Eikonal_equation#Numerical_approximation
    """
    u_i = min(
        safe_get_times_3d(times, i-1, j, k),
        safe_get_times_3d(times, i+1, j, k)
    )
    u_j = min(
        safe_get_times_3d(times, i, j-1, k),
        safe_get_times_3d(times, i, j+1, k)
    )
    u_k = min(
        safe_get_times_3d(times, i, j, k-1),
        safe_get_times_3d(times, i, j, k+1)
    )
    u_sum = u_i + u_j + u_k

    speed_val = speed_func_arr[i, j, k] + EPS
    reciproc_speed_func_elm = 1.0 / speed_val
    n = 3.0

    if not math.isinf(u_sum):
        inner_parenthes = u_i * u_i + u_j * u_j + u_k * u_k \
            - reciproc_speed_func_elm ** 2
        discriminant = u_sum*u_sum - n * inner_parenthes
        if discriminant > 0.0:
            return (u_sum + np.sqrt(discriminant)) / n
    return reciproc_speed_func_elm + min(u_i, u_j, u_k)


def get_neighbors_3d(status: CachingDataStructure,
                     i: int, j: int, k: int) -> list:
    """
    Returns a list of tuples of all coordinates that are direct neighbors,
    meaning the index is valid and they are not KNOWN
    """
    coords = []
    for (x, y, z) in [(i-1, j, k), (i+1, j, k),
                      (i, j-1, k), (i, j+1, k),
                      (i, j, k-1), (i, j, k+1)]:
        if status.valid_index(x, y, z) and not status[x, y, z]:  # Not known
            coords.append((x, y, z))
    return coords


def fmm_3d(speed_func_arr: CachingDataStructure,
           start_point: tuple,
           end_point: tuple = None) -> CachingDataStructure:
    """
    Computes the FMM for a 3D speed function and a given start point. If
    an end point is supplied, only the necessary coordinates will be computed.
    Otherwise all coordinates
    """
    i, j, k = start_point

    times = speed_func_arr.empty_of_same_shape()
    times.fill(INF)
    times[i, j, k] = 0.0

    status = times.empty_of_same_shape()
    status.fill(False)
    status[i, j, k] = KNOWN

    heap_pointers = status.empty_of_same_shape()
    heap_pointers.fill(Neighbor(0, 0, False), dtype='object')

    neighbors = []
    heapify(neighbors)

    first_run = True
    while (i, j, k) != end_point and (first_run or neighbors):
        first_run = False

        # Get new neighbors
        coords = get_neighbors_3d(status, i, j, k)

        # Calculate arrival times of newly added neighbors
        for (x, y, z) in coords:
            val_time = calc_time_3d(speed_func_arr, times, x, y, z)
            if val_time < times[x, y, z]:
                if heap_pointers[x, y, z].valid:
                    heap_pointers[x, y, z].valid = False
                times[x, y, z] = val_time
                heap_elm = Neighbor(val_time, [x, y, z], True)
                heap_pointers[x, y, z] = heap_elm
                heappush(neighbors, heap_elm)

        # Find the smallet value of neighbours
        while neighbors:
            element = heappop(neighbors)
            if element.valid:
                break
        i, j, k = element.coords

        # Add coordinate of smallest value to known
        status[i, j, k] = KNOWN
    return times


###########################
#     General case        #
###########################
def get_direct_neighbour_coords_general(key: tuple) -> list:
    """
    Gets the direct neighbors given a coordinate key. I.e. if key = (3,3,3),
    then [(2,3,3), (4,3,3), (3,2,3), (3,4,3), (3,3,2), (3,3,4)] is returned.
    """
    coords = []
    key_lst = list(key)
    for i in range(len(key)):
        key_lst[i] -= 1
        coords.append(tuple(key_lst))
        key_lst[i] += 2
        coords.append(tuple(key_lst))
        key_lst[i] -= 1
    return coords


def safe_get_times_general(times: CachingDataStructure, key: tuple) -> float:
    """
    Returns the saved time value in an n-dimensional array if is a valid index,
    otherwise infinity
    """
    return INF if not times.valid_index(*key) else times[key]


def calc_time_general(speed_func_arr: CachingDataStructure,
                      times: CachingDataStructure, key: tuple) -> float:
    """
    Calculates the time value for a newly added neighbor in n dimensions.
    Implementation based on
    https://en.wikipedia.org/wiki/Eikonal_equation#Numerical_approximation
    """
    Us = np.array([])
    neighbour_coords = get_direct_neighbour_coords_general(key)
    i = 0
    for _ in range(len(neighbour_coords) // 2):
        forwards_op = safe_get_times_general(times, neighbour_coords[i])
        backwars_op = safe_get_times_general(times, neighbour_coords[i+1])
        Us = np.append(Us, min(forwards_op, backwars_op))
        i += 2

    speed_val = speed_func_arr[key] + EPS
    reciproc_speed_func_elm = 1.0 / speed_val
    Us_sum = np.sum(Us)
    n = float(speed_func_arr.dim)

    if not math.isinf(Us_sum):
        inner_parentheses = np.sum(Us * Us) - reciproc_speed_func_elm**2
        discriminant = Us_sum - n * inner_parentheses
        if discriminant > 0.0:
            return (Us_sum + np.sqrt(discriminant)) / n
    return reciproc_speed_func_elm + np.min(Us)


def get_neighbors_general(status: CachingDataStructure, key: tuple) -> list:
    """
    Returns a list of tuples of all coordinates that are direct neighbors,
    meaning the index is valid and they are not KNOWN
    """
    coords = []
    for key in get_direct_neighbour_coords_general(key):
        if status.valid_index(*key) and not status[key]:  # Not known
            coords.append(key)
    return coords


def fmm(speed_func_arr: CachingDataStructure,
        start_point: tuple,
        end_point: tuple = None) -> CachingDataStructure:
    """
    Computes the FMM for a n-dimensional speed function and a given start point
    If an end point is supplied, only the necessary coordinates will be
    computed. Otherwise all coordinates
    """
    assert speed_func_arr.dim == len(start_point), \
        f"Difference in dimensionality of speed func ({speed_func_arr.dim}) \
          and start_point {start_point}"
    if speed_func_arr.dim == 2:
        return fmm_2d(speed_func_arr, start_point, end_point)
    elif speed_func_arr.dim == 3:
        return fmm_3d(speed_func_arr, start_point, end_point)
    # else keep doing general

    times = speed_func_arr.empty_of_same_shape()
    times.fill(INF)
    times[start_point] = 0.0

    status = times.empty_of_same_shape()
    status.fill(False)
    status[start_point] = KNOWN

    heap_pointers = status.empty_of_same_shape()
    heap_pointers.fill(Neighbor(0, 0, False), dtype='object')

    neighbors = []
    heapify(neighbors)

    first_run = True
    current_key = start_point
    while (current_key != end_point) and (first_run or neighbors):
        first_run = False

        # Get new neighbors
        coords = get_neighbors_general(status, current_key)

        # Calculate arrival times of newly added neighbors
        for key in coords:
            val_time = calc_time_general(speed_func_arr, times, key)
            if val_time < times[key]:
                if heap_pointers[key].valid:
                    heap_pointers[key].valid = False
                times[key] = val_time
                heap_elm = Neighbor(val_time, key, True)
                heap_pointers[key] = heap_elm
                heappush(neighbors, heap_elm)

        # Find the smallet value of neighbours
        while neighbors:
            element = heappop(neighbors)
            if element.valid:
                break
        current_key = element.coords

        # Add coordinate of smallest value to known
        status[current_key] = KNOWN
    return times
