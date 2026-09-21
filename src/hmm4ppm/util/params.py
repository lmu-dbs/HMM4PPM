import json
import hashlib
import os

def hash_param_dict(d):
    serialized = json.dumps(d, sort_keys=True, default=str).encode('utf-8')
    h = hashlib.sha256(serialized).hexdigest()
    return h, serialized

def save_hash_dict(h, serialized, hash_save_filepath):
    
    last_file_state = load_all(hash_save_filepath)

    if h not in last_file_state.keys():        
        with open(hash_save_filepath, 'a') as f:
            f.write(json.dumps({h: json.loads(serialized)}) + '\n')

def load_all(filepath):
    entries = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.update(json.loads(line))
    return entries