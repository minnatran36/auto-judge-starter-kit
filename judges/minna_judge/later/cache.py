import os
import json

def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
    
def save_cache(data, path):
    with open(path, "w") as f:
        json.dump(data, f)