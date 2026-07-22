import pickle
import numpy as np
import h5py
import os
import cv2 as cv

def main():
    # Keep these values consistent with the arguments used by 1_recorde.py.
    task = 'Splicing_3'
    task_name = f'task/task_{task}'
    root_folder = '/home/nova/mir'
    dataset_folder = os.path.join(root_folder, f'dataset/dataset_{task}')

    print(f"Processed dataset will be saved to: {dataset_folder}")
    create_folder(dataset_folder)

    task_folder = os.path.join(root_folder, task_name)
    epochs = len(os.listdir(task_folder))

    for epoch in range(epochs):
        epoch_folder = os.path.join(task_folder, f'epoch_{epoch}')
        action_folder = os.path.join(epoch_folder, 'Action')
        observation_folder = os.path.join(epoch_folder, 'Observations')
        image_folder = os.path.join(observation_folder, 'img')
        qpos_folder = os.path.join(epoch_folder, 'Action')
        stage_folder= os.path.join(observation_folder, 'stage')

        # Read one recorded episode.
        action = read_files(action_folder, 'pkl')
        qpos = read_files(qpos_folder, 'pkl')
        images = read_files(image_folder, 'png')
        stage = read_files(stage_folder, 'pkl')

        # Write the episode in ACT-compatible HDF5 format.
        hdf_file_name = os.path.join(dataset_folder, f'episode_{epoch}.hdf5')
        hdf5_create(hdf_file_name, action, qpos, stage, images)

def create_folder(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def count_files(directory, extension):
    """Count files with the requested extension in a directory."""
    return len([file for file in os.listdir(directory) if file.endswith(extension)])

def read_files(directory, extension):
    """Read a numerically ordered sequence of episode files."""
    merged_data = []
    file_count = count_files(directory, extension)
    for index in range(file_count):
        if extension == 'pkl':
            file_path = os.path.join(directory, f'{index}.{extension}')
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # Pad state and action vectors to the model's 14 dimensions.
                data = np.pad(data, (0, 14 - len(data)), 'constant')
                merged_data.append(data)
        elif extension == 'png':
            file_path = os.path.join(directory, f'img_{index}.{extension}')
            img = cv.imread(file_path)
            merged_data.append(img)
    return np.array(merged_data, dtype=np.float32 if extension == 'pkl' else None)

def hdf5_create(file_name, actions, qpos, stage, images):
    """Create an HDF5 episode and store actions and observations."""
    with h5py.File(file_name, 'w') as hdf:
        observations_group = hdf.create_group('observations')
        hdf.create_dataset('action', data=actions)
        observations_group.create_dataset('qpos', data=qpos)
        observations_group.create_dataset('stage', data=stage)
        images_group = observations_group.create_group('images')
        images_group.create_dataset('top', data=images)


if __name__ == '__main__':
    main()
