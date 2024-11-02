# benchmarks/test_performance.py
import time
from src.randomness.brownian_motion import brownian_motion_paths

def benchmark_paths():
    results = []
    for n_paths in [10, 100, 500, 1000]:
        start = time.time()
        _, paths = brownian_motion_paths(T=1, n_steps=100, initial_value=0, n_paths=n_paths)
        duration = time.time() - start
        results.append((n_paths, duration))
        print(f"Paths: {n_paths}, Time: {duration:.2f} seconds")

benchmark_paths()
