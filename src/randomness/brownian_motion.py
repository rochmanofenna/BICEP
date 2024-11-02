import numpy as np
import matplotlib.pyplot as plt
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import psutil
import logging
import os
from src.randomness.stochastic_control import apply_stochastic_controls

# Setup logging
logging.basicConfig(filename='/mnt/c/Users/ryanc/Desktop/BICEP/results/logs/simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_system_resources():
    """Detects system resources to set optimal parameter values."""
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_count = psutil.cpu_count(logical=False)

    try:
        import cupy as cp
        gpu_available = True
        gpu_memory_gb = cp.cuda.Device(0).mem_info[1] / (1024 ** 3)
    except ImportError:
        gpu_available = False
        gpu_memory_gb = 0

    logging.info(f"System Resources Detected - Memory: {available_memory_gb:.2f} GB, "
                 f"CPU Cores: {cpu_count}, GPU: {'Yes' if gpu_available else 'No'}, "
                 f"GPU Memory: {gpu_memory_gb:.2f} GB")
    return available_memory_gb, cpu_count, gpu_available, gpu_memory_gb

def calculate_optimal_parameters(n_paths, n_steps, available_memory_gb, cpu_count, gpu_available, gpu_memory_gb):
    estimated_path_memory = n_steps * 4 / (1024 ** 2)
    max_batch_size = int((available_memory_gb * 1024) // estimated_path_memory)
    batch_size = min(max(500, max_batch_size // 2), n_paths, cpu_count * 100)
    gpu_threshold = 1000 if gpu_available and gpu_memory_gb >= 8 else n_paths + 1
    save_interval = max(2000, batch_size * 2)

    logging.info(f"Calculated Parameters - Batch Size: {batch_size}, GPU Threshold: {gpu_threshold}, "
                 f"Save Interval: {save_interval}")
    return batch_size, save_interval, gpu_threshold

def setup_dask_cluster(n_paths, large_scale_threshold=5000):
    """Configures Dask cluster based on workload."""
    # For local testing only
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='2GB')
    client = Client(cluster)

    logging.info("Using local Dask cluster for testing.")
    return client

def simulate_single_path(T, n_steps, initial_value, dt, directional_bias, variance_adjustment, xp, apply_controls):
    # Mock values for missing parameters in apply_stochastic_controls
    state_visit_count = 1
    feedback_value = 0.5
    time_step = 0.1
    total_steps = n_steps

    time = xp.linspace(0, T, n_steps + 1)
    increments = xp.random.normal(0, 1, n_steps)
    
    if variance_adjustment:
        variance_scale = variance_adjustment(time[1:])
        increments *= xp.sqrt(variance_scale * dt)
    else:
        increments *= xp.sqrt(dt)
    
    # Apply stochastic controls to each increment
    for i in range(len(increments)):
        increments[i] = apply_controls(increments[i], state_visit_count, feedback_value, time_step, total_steps)

    path = xp.zeros(n_steps + 1)
    path[0] = initial_value
    path[1:] = xp.cumsum(increments) + initial_value
    return path


def brownian_motion_paths(T=1, n_steps=100, initial_value=0, n_paths=10, directional_bias=0.0, variance_adjustment=None):
     # Validate inputs
    if T <= 0:
        raise ValueError("Time duration T must be greater than 0.")
    if n_steps <= 0:
        raise ValueError("Number of steps n_steps must be greater than 0.")
    if n_paths < 0:
        raise ValueError("Number of paths n_paths must be non-negative.")

    if n_paths == 0:
        return np.array([]), np.empty((0, n_steps + 1))

    available_memory_gb, cpu_count, gpu_available, gpu_memory_gb = detect_system_resources()
    batch_size, save_interval, gpu_threshold = calculate_optimal_parameters(
        n_paths, n_steps, available_memory_gb, cpu_count, gpu_available, gpu_memory_gb
    )

    client = setup_dask_cluster(n_paths)
    
    try:
        if n_paths >= gpu_threshold:
            import cupy as cp
            xp = cp
            use_gpu = True
        else:
            xp = np
            use_gpu = False
    except ImportError:
        xp = np
        use_gpu = False

    dt = T / n_steps
    time = xp.linspace(0, T, n_steps + 1)
    if use_gpu:
        time = time.get()

    file_path = "brownian_paths.dat"
    if os.path.exists(file_path):
        os.remove(file_path)
    paths = np.memmap(file_path, dtype='float32', mode='w+', shape=(n_paths, n_steps + 1))

    for i in range(0, n_paths, batch_size):
        batch_end = min(i + batch_size, n_paths)
        batch_size_current = batch_end - i

        tasks = [delayed(simulate_single_path)(T, n_steps, initial_value, dt, directional_bias, variance_adjustment, xp, apply_stochastic_controls) 
                 for _ in range(batch_size_current)]
        batch_paths = compute(*tasks)

        for j, path in enumerate(batch_paths):
            paths[i + j] = path.get() if use_gpu else path

        if (i + batch_size) % save_interval == 0 or batch_end == n_paths:
            paths.flush()
            logging.info(f"Saved up to path {batch_end}/{n_paths}")

    return time, paths


# Example usage
if __name__ == "__main__":
    # Set up the Dask cluster on an available port to avoid conflicts
    cluster = LocalCluster(dashboard_address=":0")
    client = Client(cluster)

    # Call the function or perform operations that use Dask and multiprocessing
    time, brownian_paths = brownian_motion_paths(T=1, n_steps=100, initial_value=0, n_paths=10)

    plt.figure(figsize=(10, 6))
    for i in range(min(10, 10000)):
        plt.plot(time, brownian_paths[i], label=f'Path {i+1}')
    plt.title("Brownian Motion Paths (Adaptive and Distributed)")
    plt.xlabel("Time")
    plt.ylabel("W(t)")
    plt.legend()
    plt.show()

    # Optionally, close the client after execution
    client.close()
    print("Dask processing complete.")
