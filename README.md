# Caching Strategies for Image Processing

This repository contains the source code used for the bachelor thesis *Caching Strategies for Image Processing*. A number of algorithms that are interesting in a caching context have been implemented and their cache performance has been tested/compared on three different data structures.

### Algorithms
* Recursive Matrix Multiplication
* Spatial Convolution
* Fast Marching Method
* Fast Fourier Transform

### Data Structures
* Morton Ordering
* BlockArray
* Standard row major array

If you want to learn more about these, you are welcome to read the [thesis].

## Running the code

The code uses [pycachesim] for simulating the cache, so to be able to run the code you will need to:

```shell
$ pip install pycachesim
```

### 

The tests of performance itself and plots of the results are defined in the different `.ipynb`-files in the top level directory. As the performance tests can take quite a long time to run (8+ hours), the results have been saved in JSON format to the `results/`-folder.

When viewing the Notebooks, you can either simply glance over the already plotted results, load in the generated JSON and play with the data or, if you wish to generate the results anew, change the following line which is in the top of all the Notebooks:

```python
# Set to true if you want to run the tests again. Otherwise just loads results from JSON
GENERATE_NEW_RESULTS = False
```

## Testing correctness

To test that the algorithms and data structures work as intended, a number of correctness tests have been defined in the `correctness-tests/`-folder. To run these, navigate to the test folder and run `$ pytest`.


## Credits

* [Jon Sporring] for his excellent supervision with this project.
* [pycachesim] for allowing to run the cache simulations.

[thesis]: #
[pycachesim]: https://github.com/RRZE-HPC/pycachesim
[Jon Sporring]: https://github.com/sporring