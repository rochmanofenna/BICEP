import numpy as np
import logging
from functools import lru_cache

# Setup logging with adjustable verbosity
logging.basicConfig(filename='/mnt/c/Users/ryanc/Desktop/BICEP/results/logs/stochastic_control.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable thresholds (can adjust as needed)
HIGH_THRESHOLD = 10
LOW_THRESHOLD = 2
DECAY_RATE = 0.1
FEEDBACK_WARNING_THRESHOLD = 0.9  # Threshold for feedback alerts
ADAPTIVE_ADJUSTMENT = 0.05  # Degree of adaptive tuning

# Enable diagnostic logging for detailed insights
DIAGNOSTIC_LOGGING = True  # Set to False for standard logging

def log_info(message):
    """Helper function to handle diagnostic logging."""
    if DIAGNOSTIC_LOGGING:
        logging.info(message)

def adjust_variance(base_variance, adjustment_factor=1.0):
    """
    Adjusts the variance of the Brownian motion based on an adjustment factor.
    """
    adjusted_variance = base_variance * adjustment_factor
    log_info(f"Adjusted variance from {base_variance} to {adjusted_variance} using factor {adjustment_factor}")
    return adjusted_variance

@lru_cache(maxsize=128)
def adaptive_randomness_control(brownian_increment, feedback_value, target_range=(0.2, 1.0)):
    """
    Adjusts randomness based on feedback, with LRU caching for repeated patterns.
    """
    # Warn if feedback value is high
    if feedback_value >= FEEDBACK_WARNING_THRESHOLD:
        logging.warning(f"Feedback value {feedback_value} exceeds threshold {FEEDBACK_WARNING_THRESHOLD}")
    
    # Scale feedback value within the target range
    scale_factor = np.clip(0.5 + feedback_value * 0.5, *target_range)
    adjusted_increment = brownian_increment * scale_factor
    log_info(f"Adjusted randomness with scale factor {scale_factor}: {adjusted_increment}")
    return adjusted_increment

def control_randomness_by_state(state_visit_count, total_steps, high_threshold=HIGH_THRESHOLD, low_threshold=LOW_THRESHOLD):
    """
    Controls randomness levels based on the normalized frequency of state visits.
    """
    normalized_count = state_visit_count / total_steps
    
    # Adjust HIGH_THRESHOLD dynamically if feedback consistently triggers alerts
    global HIGH_THRESHOLD
    if HIGH_THRESHOLD < 15 and normalized_count > 0.8 * high_threshold:
        HIGH_THRESHOLD += ADAPTIVE_ADJUSTMENT  # Increase to preempt feedback strain
        log_info(f"Adjusted HIGH_THRESHOLD adaptively to {HIGH_THRESHOLD}")
    
    # Control randomness based on visit frequency
    if normalized_count < low_threshold:
        adjustment_factor = 1.5
    elif normalized_count > high_threshold:
        adjustment_factor = 0.5
    else:
        adjustment_factor = 1.0
    log_info(f"State visit count {state_visit_count} (normalized: {normalized_count}) yields adjustment factor {adjustment_factor}")
    return adjustment_factor

def combined_variance_control(state_visit_count, total_steps, time, base_variance=1.0, high_threshold=HIGH_THRESHOLD, decay_rate=DECAY_RATE):
    """
    Combines state-visit control and time decay for adaptive variance adjustment.
    """
    state_factor = control_randomness_by_state(state_visit_count, total_steps, high_threshold=high_threshold)
    time_factor = np.exp(-decay_rate * time)
    combined_variance = base_variance * state_factor * time_factor
    log_info(f"Combined variance control: {combined_variance} for normalized visit count and time {time}")
    return combined_variance

def apply_stochastic_controls(brownian_increment, state_visit_count, feedback_value, time, total_steps, base_variance=1.0):
    """
    Integrates all stochastic controls to provide a final adjustment for the Brownian increment.
    """
    # Apply adaptive variance control based on both state and time decay
    variance_factor = combined_variance_control(state_visit_count, total_steps, time, base_variance)
    
    # Apply cached adaptive randomness control based on feedback
    feedback_adjusted_increment = adaptive_randomness_control(brownian_increment, feedback_value)
    
    # Combine adjustments for final increment
    final_increment = feedback_adjusted_increment * variance_factor
    log_info(f"Final adjusted increment: {final_increment} after applying stochastic controls")
    return final_increment
