# src/utils/config_loader.py
import json
import os

def load_config(config_path='config.json'):
    """Loads the configuration from a JSON file."""
    try:
        # Construct path relative to the project root if needed
        # This assumes config.json is in the project root
        # and this script might be called from various places.
        # For simplicity, if config_loader.py is in src/utils, 
        # and main scripts are in root or src, this path needs care.
        # A common way is to determine project root and join.
        # For now, let's assume config_path is relative to where Python is run or absolute.

        # If config.json is in the project root, and you run scripts from the root:
        # config_file = config_path 

        # A more robust way if this util is called from different depths:
        # Get the directory of the currently running script (or main script)
        # and then navigate to the project root to find config.json.
        # For now, let's keep it simple and assume config.json can be found directly
        # or the user provides a full path.

        if not os.path.exists(config_path):
            # Try to find it from a common structure relative to this file
            # if this file is in src/utils/
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root_config_path = os.path.join(script_dir, '..', '..', config_path)
            if os.path.exists(project_root_config_path):
                config_path = project_root_config_path
            else: # Fallback to current working directory if not found via relative path
                if os.path.exists(os.path.join(os.getcwd(), config_path)):
                     config_path = os.path.join(os.getcwd(), config_path)
                else:
                    print(f"Warning: config.json not found at initial path or typical project root location.")
                    # A better solution would be to pass an absolute path or use environment variables

        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading config: {e}")
        return None

# Example of how to load it (you'd do this in other files):
# if __name__ == '__main__':
#     config = load_config() # Assumes config.json is in the same dir or found via logic
#     if config:
#         print("Config loaded successfully!")
#         print(f"D1 CSV Path: {config.get('data_settings', {}).get('d1_csv_path')}")