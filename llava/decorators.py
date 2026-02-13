import time
import functools

def measuretime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            print(f"{func.__name__} took {dt:.6f}s")
    return wrapper