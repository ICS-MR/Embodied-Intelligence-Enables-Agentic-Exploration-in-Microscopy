import os
import pickle
import numpy as np

# Recording root.
base_path = "/home/nova/mir/task/task_Splicing_3"

# Inspect every epoch directory.
for epoch_dir in sorted(os.listdir(base_path)):
    if not epoch_dir.startswith("epoch_"):
        continue
    
    epoch_path = os.path.join(base_path, epoch_dir, "Action")
    if not os.path.exists(epoch_path):
        print(f"Warning: {epoch_path} does not exist; skipping")
        continue
    
    # Collect per-file means.
    epoch_means = []
    
    # Read every pickle in the Action directory.
    for pkl_file in sorted(os.listdir(epoch_path)):
        if not pkl_file.endswith(".pkl"):
            continue
        
        file_path = os.path.join(epoch_path, pkl_file)
        
        # Load one action.
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Require an [x, y, z] vector.
        if not isinstance(data, (list, np.ndarray)) or len(data) != 3:
            print(f"Warning: {file_path} is not an [x, y, z] vector; skipping")
            continue
        
        # Compute the file mean.
        mean_values = np.mean(data, axis=0) if isinstance(data, np.ndarray) else np.mean(data)
        epoch_means.append(mean_values)
    
    # Compute the epoch mean.
    if epoch_means:
        overall_mean = np.mean(epoch_means, axis=0)
        # if overall_mean > 10000 or overall_mean < -10000:
        #     print(f"{epoch_dir}: mean = {overall_mean}")
        print(f"{epoch_dir}: mean = {overall_mean}")
    else:
        print(f"{epoch_dir}: no valid data")
