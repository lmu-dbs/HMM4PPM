import os
import csv
import uuid
from datetime import datetime

def log_to_csv(csv_file, params_metrics):
    file_exists = os.path.isfile(csv_file)
    
    row = {
        "run_id": str(uuid.uuid4()),
        "finish_time": datetime.now().isoformat(),
    }
    
    for key, value in params_metrics.items():
        row[key] = value

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)