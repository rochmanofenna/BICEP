import unittest
from src.randomness.brownian_motion import (
    detect_system_resources,
    calculate_optimal_parameters,
    simulate_single_path,
    brownian_motion_paths
)

class TestBrownianMotion(unittest.TestCase):
    def test_detect_system_resources(self):
        resources = detect_system_resources()
        self.assertTrue(isinstance(resources, tuple))
        self.assertEqual(len(resources), 4)  # Expected: (memory, CPU count, GPU availability, GPU memory)

    def test_calculate_optimal_parameters(self):
        resources = (8, 4, True, 16)
        params = calculate_optimal_parameters(1000, 100, *resources)
        self.assertTrue(isinstance(params, tuple))

    def test_simulate_single_path(self):
        path = simulate_single_path(1, 100, 0, 0.01, 0, None, np)
        self.assertEqual(len(path), 101)

    def test_brownian_motion_paths(self):
        time, paths = brownian_motion_paths(1, 100, 0, 10)
        self.assertEqual(len(time), 101)
        self.assertEqual(paths.shape, (10, 101))

if __name__ == "__main__":
    unittest.main()
