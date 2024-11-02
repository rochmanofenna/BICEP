from src.randomness.brownian_motion import detect_system_resources
from src.randomness.stochastic_control import adjust_variance

def test_logging():
    # Call a function to generate log entries
    detect_system_resources()  # Should log system resource information
    adjust_variance(1.0, 1.2)  # Should log variance adjustment details

if __name__ == "__main__":
    test_logging()
    print("Log test complete. Check the log files in /mnt/c/Users/ryanc/Desktop/BICEP/results/logs/")
