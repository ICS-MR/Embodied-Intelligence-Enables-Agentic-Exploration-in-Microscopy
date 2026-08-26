# import h5py

# with h5py.File('/home/nova/mir/dataset/dataset_Splicing_2/episode_124.hdf5', 'r') as f:
#     actions = f['/action'][:200]  # Take the first 20 frames.
#     print("Example actions:")
#     for i, a in enumerate(actions):
#         print(f"{i}: {a}")
import h5py
import numpy as np
import os

# Set the dataset directory path.
dataset_dir = '/home/nova/mir/dataset/dataset_Splicing_2/'

# Get all HDF5 files in the directory.
episode_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
episode_files.sort()  # Sort by filename.

print(f"Found {len(episode_files)} episode files")

# Iterate over each episode file.
for episode_file in episode_files:
    file_path = os.path.join(dataset_dir, episode_file)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check whether the action dataset exists.
            if 'action' not in f:
                print(f"{episode_file}: action dataset not found")
                continue
                
            actions = f['/action'][:]  # Read all action data.
            action_mean = np.mean(actions, axis=0)  # Calculate the mean.
            
            print(f"{episode_file}: action mean = {action_mean}")
            
    except Exception as e:
        print(f"Error processing {episode_file}: {str(e)}")
