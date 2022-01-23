import os
import json
import os 

# TODO Add experiment config specific functionality if necessary (e.g. resolve classes)
def load_experiment_config(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not locate experiment config file '{filepath}'")
    with open(filepath, "r") as json_file:
        return json.load(json_file)
