import os
import pickle
import numpy as np

# Base path
base_path = "/home/nova/mir/task/task_Push_to_target"

# Iterate over each epoch folder.
for epoch_dir in sorted(os.listdir(base_path)):
    if not epoch_dir.startswith("epoch_"):
        continue
    
    epoch_path = os.path.join(base_path, epoch_dir, "Action")
    if not os.path.exists(epoch_path):
        print(f"Warning: {epoch_path} does not exist; skipping")
        continue
    
    # Collect the mean from each PKL file.
    epoch_means = []
    
    # Iterate over all PKL files in the Action directory.
    for pkl_file in sorted(os.listdir(epoch_path)):
        if not pkl_file.endswith(".pkl"):
            continue
        
        file_path = os.path.join(epoch_path, pkl_file)
        
        # Load the PKL file.
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Ensure that the data uses the [x, y, z] format.
        if not isinstance(data, (list, np.ndarray)) or len(data) != 3:
            print(f"Warning: {file_path} does not use the [x, y, z] format; skipping")
            continue
        
        # Calculate the mean.
        mean_values = np.mean(data, axis=0) if isinstance(data, np.ndarray) else np.mean(data)
        epoch_means.append(mean_values)
    
    # Calculate the mean for the entire epoch.
    if epoch_means:
        overall_mean = np.mean(epoch_means, axis=0)
        # if overall_mean > 10000 or overall_mean < -10000:
        #     print(f"{epoch_dir}: mean = {overall_mean}")
        print(f"{epoch_dir}: mean = {overall_mean}")
    else:
        print(f"{epoch_dir}: no valid data")
