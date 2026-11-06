
import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config_loader import load_config

def main():
    config = load_config('phase_0')
    print("Starting Phase 0: Sensor Health")
    print(f"UDP Port: {config.get('udp_port')}")
    print(f"Log Duration: {config.get('log_duration_sec')}s")

if __name__ == '__main__':
    main()
