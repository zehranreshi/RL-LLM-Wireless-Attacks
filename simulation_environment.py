import math
import time

# The optimal target values
TARGET_FREQ = 3.6192e9
TARGET_BW = 20e6
TARGET_GAIN = 85.0

def mock_run_simulation_and_get_reward(config: dict) -> float:
    print(f"--- Running Simulation ---")
    print(f"Config: {config}")

    try:
        center_freq = float(config.get("center_frequency", 0))
        bandwidth = float(config.get("bandwidth", 0))
        tx_gain = float(config.get("tx_gain", 0))
    except (TypeError, ValueError):
        print("Invalid config format. Reward is -0.1.")
        return -0.1

    # Use the globally set TARGET_FREQ for this episode's calculation
    global TARGET_FREQ
    
    freq_diff = abs(center_freq - TARGET_FREQ)
    max_reasonable_diff = 100e6
    freq_score = max(0.0, 1.0 - (freq_diff / max_reasonable_diff))

    bw_diff = abs(bandwidth - TARGET_BW)
    bw_score = max(0.0, 1.0 - (bw_diff / (TARGET_BW * 2)))

    gain_diff = abs(tx_gain - TARGET_GAIN)
    gain_score = max(0.0, 1.0 - (gain_diff / 40.0))

    final_reward = (freq_score * 0.6) + (bw_score * 0.2) + (gain_score * 0.2)
    
    pdr = 1.0 - final_reward 
    
    if final_reward > 0.95:
        final_reward = 1.0

    print(f"Scores -> Freq: {freq_score:.2f}, BW: {bw_score:.2f}, Gain: {gain_score:.2f}")
    print(f"Result -> PDR: {pdr:.3f}, Reward: {final_reward:.3f}")
    print(f"--------------------------\n")
    
    return final_reward