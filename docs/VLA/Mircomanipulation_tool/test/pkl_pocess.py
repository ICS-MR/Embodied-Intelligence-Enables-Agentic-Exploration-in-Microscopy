import pickle
import glob
import os
import numpy as np

def modify_z_axis(file_path):
    """Set the Z component of an [x, y, z] pickle value to zero."""
    try:
        # Read the pickle.
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        is_modified = False

        # Handle an [x, y, z] list.
        if isinstance(data, list) and len(data) >= 3:
            # Avoid rewriting values that are already zero.
            if data[2] != 0:
                data[2] = 0
                is_modified = True
        
        # Handle a NumPy robot-state vector.
        elif isinstance(data, np.ndarray) and data.size >= 3:
            # Process a one-dimensional [x, y, z] array.
            if data.ndim == 1:
                if data[2] != 0:
                    data[2] = 0
                    is_modified = True
            # Enable this branch to process a two-dimensional trajectory.
            # elif data.ndim == 2 and data.shape[1] >= 3:
            #     data[:, 2] = 0
            #     is_modified = True

        # Write the pickle only when its value changed.
        if is_modified:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[MODIFIED] {file_path}")
        else:
            # The format is unsupported or Z is already zero.
            # print(f"[SKIPPED] {file_path}")
            pass

    except Exception as e:
        print(f"[ERROR] Unable to process {file_path}: {e}")

def main():
    base_path = "/home/nova/mir/task/task_Splicing_3"
    
    # File matching rules.
    
    # Match Action pickles from every epoch.
    action_pattern = os.path.join(base_path, "epoch_*/Action/*.pkl")
    
    # Match qpos pickles from every epoch.
    qpos_pattern = os.path.join(base_path, "epoch_*/Observations/qpos/*.pkl")
    
    # Collect matching files.
    files_to_process = []
    files_to_process.extend(glob.glob(action_pattern))
    files_to_process.extend(glob.glob(qpos_pattern))
    
    print(f"Found {len(files_to_process)} files to process...")
    
    # Process the batch.
    for file_path in files_to_process:
        modify_z_axis(file_path)

    print("Processing complete.")

if __name__ == "__main__":
    main()
