# calculate fib number with timestamp
from time import time


def fib_with_time(n):
    start = time()
    result = fib(n)
    end = time()
    print("fib({}) = {} (time: {})".format(n, result, end-start))
    return result

def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n-1) + fib(n-2)

fib_with_time(40)
