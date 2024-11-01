import unittest
from src.randomness.stochastic_control import (
    adjust_variance,
    adaptive_randomness_control,
    control_randomness_by_state,
    combined_variance_control,
    apply_stochastic_controls
)

class TestStochasticControl(unittest.TestCase):
    def test_adjust_variance(self):
        adjusted = adjust_variance(1.0, 1.2)
        self.assertEqual(adjusted, 1.2)

    def test_adaptive_randomness_control(self):
        increment = adaptive_randomness_control(0.1, 0.8)
        self.assertTrue(0.05 <= increment <= 0.1)

    def test_control_randomness_by_state(self):
        factor = control_randomness_by_state(5, 20)
        self.assertTrue(factor > 0)

    def test_combined_variance_control(self):
        variance = combined_variance_control(5, 20, 0.5)
        self.assertTrue(variance >= 0)

    def test_apply_stochastic_controls(self):
        final_increment = apply_stochastic_controls(0.1, 5, 0.8, 0.5, 20)
        self.assertTrue(final_increment > 0)

if __name__ == "__main__":
    unittest.main()
