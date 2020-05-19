import numpy as np
from itertools import product
from typing import Any

from data_structures.caching_data_stucture import CachingDataStructure


def tuple_gen(new_val: Any, ori_tuple: tuple, axis: int) -> tuple:
    """
    Returns a tuple similar to ori_tuple but with new_val at a specifix axis
    """
    lst = list(ori_tuple)
    lst[axis] = new_val
    return tuple(lst)


def rev_bit_str(n: int, bitlen: int) -> int:
    """
    Reverses the bitstring of length bitlen of a number n and returns the
    bitreverses integer. I.e. rev_bits(3, 4) => "0011" => "1100" => 12
    """
    bits = bin(n)[2:]  # Truncate first 2 characters to avoid '0b'
    start_pow = bitlen - len(bits)
    return sum([2 ** (idx + start_pow) if bit == '1' else 0
                for idx, bit in enumerate(bits)])


def bit_rev_inplace(arr: CachingDataStructure, axis: int) \
                    -> CachingDataStructure:
    """
    Swaps the values along a given axis according to the reversed bit value
    of that index
    """
    N = arr.shape[0]
    bitlen = int(np.log2(N))
    for key in arr.iter_keys():
        rev_bit_idx = rev_bit_str(key[axis], bitlen)
        if key[axis] >= rev_bit_idx:  # Make sure to only swap elements once
            continue
        new_key = tuple_gen(rev_bit_idx, key, axis)
        arr[key], arr[new_key] = arr[new_key], arr[key]
    return arr


def iter_fft_1d(output: CachingDataStructure, N: int, idxs: tuple, axis: int) \
                -> None:
    """
    Performs the FFT on an n-Dimensional CachingDataStructure along a specific
    1-dimensional axis. output should be the bit-reversed signal array and
    idxs is a tuple of indices that should be held constant except for the one
    at axis. Doesn't return anything but output along the given axis has now
    undergone FFT. Assumes the output-array has already been bit-reversed.
    """
    for s in range(1, int(np.log2(N)) + 1):
        m = 2**s
        w_m = np.exp(-2j * np.pi / m)
        for k in range(0, N, m):
            w = 1
            for j in range(m//2):
                key1 = tuple_gen(k + j + m // 2, idxs, axis)
                key2 = tuple_gen(k + j, idxs, axis)

                t = w * output[key1]
                u = output[key2]
                output[key2] = u + t
                output[key1] = u - t
                w *= w_m


def fftn(signal: CachingDataStructure) -> CachingDataStructure:
    """
    Performs FFT on an n-dimensional input signal. The input should be a
    perfect square/cube/etc. and have a side length in each dimension of 2**i
    where i is some integer. It is done inplace so original array is
    modified.
    """
    N, dim = signal.shape[0], signal.dim
    for axis in range(dim - 1, -1, -1):  # i.e. [2, 1, 0] for dim = 3
        bit_rev_inplace(signal, axis=axis)
        iter_ranges = ([None] if axis == i else range(N) for i in range(dim))
        for idxs in product(*iter_ranges):
            iter_fft_1d(signal, N, idxs, axis)
    return signal


def bit_rev_copy(arr: CachingDataStructure, axis: int) -> CachingDataStructure:
    """
    Copies an CachingDataStructure, but swaps the values along a given axis
    according to the reversed bit value of that index
    """
    rev_arr = arr.empty_of_same_shape().fill(complex(0, 0), dtype='complex')
    n = arr.shape[0]
    bitlen = int(np.log2(n))
    for key in rev_arr.iter_keys():
        new_key = tuple_gen(rev_bit_str(key[axis], bitlen), key, axis)
        rev_arr[new_key] = arr[key]
    return rev_arr


def fftn_copy_arr(signal: CachingDataStructure) -> CachingDataStructure:
    """
    Performs FFT on an n-dimensional input signal. The input should be a
    perfect square/cube/etc. and have a side length in each dimension of 2**i
    where i is some integer. It is done on a number of copys on the orignal
    signal array.
    """
    N, dim = signal.shape[0], signal.dim
    prev = signal
    for axis in range(dim - 1, -1, -1):  # i.e. [2, 1, 0] for dim = 3
        rev_signal = bit_rev_copy(prev, axis=axis)
        iter_ranges = ([None] if axis == i else range(N) for i in range(dim))
        for idxs in product(*iter_ranges):
            iter_fft_1d(rev_signal, N, idxs, axis)
        prev = rev_signal
    return rev_signal
