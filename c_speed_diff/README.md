# Testing the speed difference between C and Python

The code here does a quick test of the speed difference between C and Python. To run the code, first `make` the `funcs.so` file:

```
$ make
```

Afterwards, run the following to actually test the speed difference and write the results to the file `results.txt`:

```
$ python speed_diff.py
```

My `results.txt` file has been included so the code doesn't need to get rerun to see the results. A speed difference of about 50x was observed.