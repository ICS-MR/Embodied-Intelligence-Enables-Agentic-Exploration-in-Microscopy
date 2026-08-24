import pickle
import glob
import os
import numpy as np  # Import NumPy in case the data is stored as an array.

def modify_z_axis(file_path):
    """
    Read a PKL file, set the Z value in [x, y, z] data to 0, and save it.
    """
    try:
        # 1. Read the data.
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        is_modified = False

        # 2. Inspect and modify the data, assuming a list or NumPy array.
        # Case A: the data is a list [x, y, z].
        if isinstance(data, list) and len(data) >= 3:
            # Modify only when Z is nonzero to avoid unnecessary writes.
            if data[2] != 0:
                data[2] = 0
                is_modified = True
        
        # Case B: the data is a NumPy array, which is common in robot data.
        elif isinstance(data, np.ndarray) and data.size >= 3:
             # Handle a one-dimensional [x, y, z] array.
            if data.ndim == 1:
                if data[2] != 0:
                    data[2] = 0
                    is_modified = True
            # Uncomment the following block for a 2D trajectory such as [[x, y, z], ...].
            # elif data.ndim == 2 and data.shape[1] >= 3:
            #     data[:, 2] = 0
            #     is_modified = True

        # 3. Write the file back if it was modified.
        if is_modified:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[MODIFIED] {file_path}")
        else:
            # The data format is unsupported or Z is already 0.
            # print(f"[SKIPPED] {file_path} (no change needed or unsupported format)")
            pass

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")

def main():
    base_path = "/home/nova/mir/task/task_Splicing_3"
    
    # --- Path matching rules ---
    
    # Rule 1: match Action files under every epoch_{i}.
    # Use the * wildcard to match any epoch number.
    action_pattern = os.path.join(base_path, "epoch_*/Action/*.pkl")
    
    # Rule 2: match Observations/qpos under epoch_0 only.
    # To match qpos under every epoch, change 'epoch_0' to 'epoch_*'.
    qpos_pattern = os.path.join(base_path, "epoch_*/Observations/qpos/*.pkl")
    
    # Collect all matching files.
    files_to_process = []
    files_to_process.extend(glob.glob(action_pattern))
    files_to_process.extend(glob.glob(qpos_pattern))
    
    print(f"Found {len(files_to_process)} files. Preparing to process them...")
    
    # --- Start batch processing ---
    for file_path in files_to_process:
        modify_z_axis(file_path)

    print("Processing complete.")

if __name__ == "__main__":
    main()
