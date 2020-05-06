#include <stdio.h>

int fib(int n) {
    return n <= 1 ? 1 : fib(n-2) + fib(n-1);
}
