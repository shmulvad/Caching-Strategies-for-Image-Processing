import time
from ctypes import *

# Number of iterations to run recursive fib function for
ITERS = 35

# Load C functions
so_file = "./funcs.so"
funcs = CDLL(so_file)

# Define native fib Python function
fibs_native = lambda n: 1 if n <= 1 else fibs_native(n-1) + fibs_native(n-2)

# Test that the two functions return the same and that they return the correct value
assert funcs.fib(5) == fibs_native(5)
assert funcs.fib(12) == fibs_native(12)
assert fibs_native(12) == 233


# Calling through C
start = time.time()
for i in range(ITERS):
    funcs.fib(i)
c_time = time.time() - start


# Native Python
start = time.time()
for i in range(ITERS):
    fibs_native(i)
python_time = time.time() - start


# Write results to file
with open("results.txt", "w") as f:
    f.write(f'Number of iters: {ITERS}\n\n')

    f.write(f'C:      {c_time:.4f} s\n')
    f.write(f'Python: {python_time:.4f} s\n\n')

    f.write(f'C was {python_time / c_time:.1f}x faster')
