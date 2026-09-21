def _get_pattern_center_index(pattern: dict) -> int:
    center_index = int(len(pattern['name'].split(','))/2)
    return center_index

def _get_pattern_center(pattern: dict) -> int:
    center_index = _get_pattern_center_index(pattern)
    pattern_center = int(pattern['name'].split(',')[center_index])
    
    return pattern_center

def _child_matches_with_sequence(child: dict, sequence: list[int]) -> bool:
    child_relevant = [int(act) for act in child['name'].split(',')[:_get_pattern_center_index(child) + 1]]
    if len(child_relevant) > len(sequence):
        child_relevant = child_relevant[-(len(sequence)):]
    sequence_relevant = sequence[-len(child_relevant):]
    match = child_relevant==sequence_relevant
    return match

def _filter_start_end(sequence: list[int], start_activity: int, end_activity: int, only_idx: bool = False, include_last_start: bool = True, include_first_end: bool = True) -> list[int]:
    try:
        last_start_idx = len(sequence) - 1 - [True if el==start_activity else False for el in sequence[::-1]].index(True)
    except ValueError:
        last_start_idx = 0

    try:
        first_end_idx = [True if el==end_activity else False for el in sequence].index(True)
    except ValueError:
        first_end_idx = len(sequence) - 1

    indices = range(last_start_idx + int(not include_last_start), first_end_idx+1 - int(not include_first_end))
    if only_idx:
        return indices
 
    filtered_sequence = [sequence[idx] for idx in indices]

    return filtered_sequence