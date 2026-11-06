import yaml
import os

def load_config(phase_name):
    """
    Loads the YAML configuration file and merges the common constants
    with the phase-specific constants.
    
    Args:
        phase_name (str): The name of the phase (e.g., 'phase_0', 'phase_1').
        
    Returns:
        dict: A dictionary containing both common and phase-specific configurations.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.yaml')
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        
    # Get common configuration
    merged_config = full_config.get('common', {}).copy()
    
    # Get phase specific configuration and merge
    phase_config = full_config.get(phase_name, {})
    merged_config.update(phase_config)
    
    return merged_config

# Example Usage:
# if __name__ == "__main__":
#     config_phase_1 = load_config('phase_1')
#     print("UDP Port (Common):", config_phase_1['udp_port'])
#     print("Calibration Samples (Phase 1):", config_phase_1['calibration_samples'])
