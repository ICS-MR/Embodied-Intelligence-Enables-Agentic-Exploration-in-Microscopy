# import h5py

# with h5py.File('/home/nova/mir/dataset/dataset_Splicing_2/episode_124.hdf5', 'r') as f:
#     actions = f['/action'][:200]  # Read the first 200 frames.
#     print("Example actions:")
#     for i, a in enumerate(actions):
#         print(f"{i}: {a}")
import h5py
import numpy as np
import os

# Dataset directory.
dataset_dir = '/home/nova/mir/dataset/dataset_Splicing_2/'

# Find all HDF5 episodes.
episode_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
episode_files.sort()

print(f"Found {len(episode_files)} episode files")

# Inspect each episode.
for episode_file in episode_files:
    file_path = os.path.join(dataset_dir, episode_file)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Verify that the action dataset exists.
            if 'action' not in f:
                print(f"{episode_file}: action dataset not found")
                continue
                
            actions = f['/action'][:]
            action_mean = np.mean(actions, axis=0)
            
            print(f"{episode_file}: mean action = {action_mean}")
            
    except Exception as e:
        print(f"Error processing {episode_file}: {str(e)}")
