from itertools import product

def param_combinations(model_config):
    keys = model_config.keys()
    values = []
    
    for v in model_config.values():
        if isinstance(v, dict):
            values.append(list(param_combinations(v)))
        else:
            values.append(v)
    
    for combination in product(*values):
        result = {}
        for k, v in zip(keys, combination):
            if isinstance(v, dict):
                result[k] = v
            else:
                result[k] = v
        yield result