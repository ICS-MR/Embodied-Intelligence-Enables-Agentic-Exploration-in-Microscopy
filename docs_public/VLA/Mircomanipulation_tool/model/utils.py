import torch
import numpy as np
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.chunk_size = chunk_size
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            chunk_size = self.chunk_size
            episode_len = root['/action'].shape[0]

            # The target action window is a_{t+1:t+chunk}.
            if sample_full_episode:
                start_ts = 0
            else:
                max_start = episode_len - chunk_size - 1
                start_ts = np.random.randint(0, max_start + 1)

            qpos = root['/observations/qpos'][start_ts]
            image_dict = {
                cam_name: root[f'/observations/images/{cam_name}'][start_ts]
                for cam_name in self.camera_names
            }

            action = root['/action'][start_ts + 1 : start_ts + 1 + chunk_size]
            is_pad = np.zeros(chunk_size, dtype=np.bool_)

        all_cam_images = [image_dict[cam] for cam in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images).permute(0, 3, 1, 2).float() / 255.0

        action = action[:, :]
        action_data = torch.from_numpy(action).float()
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        is_pad = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)

    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos,}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size):
    print(f'\nData from: {dataset_dir}\n')
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def set_seed(seed): 
    torch.manual_seed(seed)
    np.random.seed(seed)


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d
