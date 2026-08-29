import pickle
import numpy as np
import h5py
import os
import cv2 as cv
import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert recorded episodes to ACT HDF5 datasets.")
    parser.add_argument('--task', type=str, default='Push_to_target', help='task suffix, for example Push_to_target')
    parser.add_argument('--task_name', type=str, default=None, help='recorded task folder name under root_folder, for example task_Cell_set_z_none')
    parser.add_argument('--root_folder', type=str, default='/home/nova/mir', help='parent directory that contains the recorded task folder')
    parser.add_argument('--dataset_folder', type=str, default=None, help='output HDF5 dataset folder')
    parser.add_argument('--compress', action='store_true', help='store images as padded JPEG bytes with /compress_len')
    parser.add_argument('--jpeg_quality', type=int, default=50, help='JPEG quality used when --compress is enabled')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    task = args.task
    root_folder = args.root_folder
    dataset_folder = args.dataset_folder or os.path.join(root_folder, f'dataset/dataset_{task}')

    print(f"Processed data will be saved to: {dataset_folder}")
    if args.compress:
        print(f"Image compression enabled: JPEG quality={args.jpeg_quality}")
    create_folder(dataset_folder)

    task_name = args.task_name or f'task_{task}'
    task_folder = os.path.join(root_folder, task_name)
    if not os.path.isdir(task_folder):
        raise FileNotFoundError(
            f"Recorded task folder not found: {task_folder}. "
            f"Set --root_folder to the parent directory that contains {task_name}."
        )
    epochs = len(os.listdir(task_folder))

    for epoch in range(epochs):
        epoch_folder = os.path.join(task_folder, f'epoch_{epoch}')
        action_folder = os.path.join(epoch_folder, 'Action')
        observation_folder = os.path.join(epoch_folder, 'Observations')
        image_folder = os.path.join(observation_folder, 'img')
        qpos_folder = os.path.join(observation_folder, 'qpos')

        # Read data.
        action = read_files(action_folder, 'pkl')
        qpos = read_files(qpos_folder, 'pkl')
        images = read_files(image_folder, 'png')

        # Create the HDF5 file.
        hdf_file_name = os.path.join(dataset_folder, f'episode_{epoch}.hdf5')
        hdf5_create(
            hdf_file_name,
            action,
            qpos,
            images,
            compress=args.compress,
            jpeg_quality=args.jpeg_quality,
        )

def create_folder(path):
    """Create the folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def count_files(directory, extension):
    """Return the number of files with the specified extension in a folder."""
    return len([file for file in os.listdir(directory) if file.endswith(extension)])

def read_files(directory, extension):
    """Read files from the specified folder and return their data."""
    merged_data = []
    file_count = count_files(directory, extension)
    for index in range(file_count):
        if extension == 'pkl':
            file_path = os.path.join(directory, f'{index}.{extension}')
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # Pad the data to a length of 14.
                data = np.pad(data, (0, 14 - len(data)), 'constant')
                merged_data.append(data)
        elif extension == 'png':
            file_path = os.path.join(directory, f'img_{index}.{extension}')
            img = cv.imread(file_path)
            if img is None:
                raise FileNotFoundError(f"Unable to read image: {file_path}")
            merged_data.append(img)
    return np.array(merged_data, dtype=np.float32 if extension == 'pkl' else None)


def encode_images_as_jpeg(images, jpeg_quality):
    """Encode one episode of images as ACT/ALOHA-style padded JPEG bytes."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality must be between 1 and 100, got {jpeg_quality}")

    encoded_images = []
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    for index, image in enumerate(images):
        success, encoded_image = cv.imencode('.jpg', image, encode_param)
        if not success:
            raise ValueError(f"JPEG encoding failed for image index {index}")
        encoded_images.append(encoded_image.reshape(-1))

    if not encoded_images:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros((0,), dtype=np.int32)

    compressed_lengths = np.array([len(image) for image in encoded_images], dtype=np.int32)
    padded_images = np.zeros((len(encoded_images), int(compressed_lengths.max())), dtype=np.uint8)
    for index, encoded_image in enumerate(encoded_images):
        padded_images[index, : len(encoded_image)] = encoded_image
    return padded_images, compressed_lengths


def hdf5_create(file_name, actions, qpos, images, compress=False, jpeg_quality=50):
    """Create an HDF5 file and save the data."""
    with h5py.File(file_name, 'w') as hdf:
        hdf.attrs['sim'] = False
        hdf.attrs['compress'] = bool(compress)
        observations_group = hdf.create_group('observations')
        hdf.create_dataset('action', data=actions)
        observations_group.create_dataset('qpos', data=qpos)
        images_group = observations_group.create_group('images')
        if compress:
            compressed_images, compressed_lengths = encode_images_as_jpeg(images, jpeg_quality)
            images_group.create_dataset('top', data=compressed_images)
            hdf.create_dataset('compress_len', data=np.array([compressed_lengths], dtype=np.int32))
        else:
            images_group.create_dataset('top', data=images)


if __name__ == '__main__':
    main()
