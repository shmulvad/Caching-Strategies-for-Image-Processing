## Initializing a CachingDataStructure
Examples below are shown for `MortonOrder`, but it is the same syntax for `RowMajorArray` and `BlockArray`.

```python
>>> import numpy as np
>>> from data_structures.morton_order import MortonOrder
>>> from data_structures.block_array import BlockArray
>>> from data_structures.row_major_array import RowMajorArray
```

#### From `numpy.ndarray`:

```python
>>> ndarray = np.random.rand(2, 2)
>>> ndarray
array([[0.08848329, 0.3314824 ],
       [0.83153481, 0.11288104]])
>>> morton = MortonOrder(ndarray)
>>> morton
morton([[0.08848329, 0.3314824 ],
        [0.83153481, 0.11288104]])
```

#### From shape as a tuple:
```python
>>> morton1 = MortonOrder(shape=(2, 2))
>>> morton1[1, 1] = 5.5
>>> morton1
morton([[0., 0. ],
        [0., 5.5]])
```

#### From another `CachingDataStructure`, but filled with a specific value
```python
>>> morton2 = morton1.empty_of_same_shape().fill(5, dtype='int')
>>> morton2
morton([[5, 5],
        [5, 5]])
```

#### With a cache starting from a given offset (can be used with all methods above)
```python
>>> # Init cache and offset variables
>>> data_with_sim_cache = MortonOrder(shape=(2, 2), cache=cache, offset=offset)
```

#### Constraints
For simplicity's sake, all `CachingDataStructure`s should be hypercubes and have a side length that is a power of 2. If you try to initialize one not fulfilling this, you'll get an AssertionError:

```python
>>> MortonOrder(shape=(4, 4, 8))
AssertionError: MortonOrder should be a hypercube but got the shape (4, 4, 8)
>>> MortonOrder(shape=(3, 3, 3))
AssertionError: MortonOrder's side length should be a power of 2 but got 3
```

## Helpful class methods

#### `valid_index(*args: int, pad: int = 0) -> bool`:
Returns`True` if the index specified is valid in this data structure and within optional padding value `pad`. Example usage:

```python
>>> morton = MortonOrder(shape=(4, 4, 4))
>>> morton.valid_index(-1, 0, 0)
False
>>> morton.valid_index(0, 0, 0)
True
>>> morton.valid_index(2, 1, 3)
True
>>> morton.valid_index(0, 0, 0, pad=1)
False
```


#### `internal_index(*args: int) -> int`:
Computes the internal 1D index at a given index in `CachingDataStructure`. Example usage:

```python
>>> shape = (128, 128, 128)
>>> MortonOrder(shape=shape).internal_index(1, 1, 1)
7
>>> BlockArray(shape=shape).internal_index(1, 1, 1)
73
>>> RowMajorArray(shape=shape).internal_index(1, 1, 1)
16513
```

#### `iter_keys() -> Generator[tuple, None, None]`
Returns a generator that yields tuples of the keys in internal linear layout (optimal spatial locality). Example usage:

```python
>>> morton = MortonOrder(np.random.rand(4, 4))
>>> list(morton.iter_keys())
[(0, 0), (1, 0), (0, 1), (1, 1),
 (2, 0), (3, 0), (2, 1), (3, 1),
 (0, 2), (1, 2), (0, 3), (1, 3),
 (2, 2), (3, 2), (2, 3), (3, 3)]
>>> for key in morton.iter_keys(): # Setting all values to 0 or 1
>>> 	morton[key] = 1.0 if morton[key] > 0.5 else 0.0
```

#### `to_numpy() -> np.ndarray`:
Transform the `CachingDataStructure` into the equivalent `np.ndarray`. Example usage:

```python
>>> MortonOrder([[1, 2], [3, 4]]).to_numpy()
array([[1, 2],
       [3, 4]])
```
