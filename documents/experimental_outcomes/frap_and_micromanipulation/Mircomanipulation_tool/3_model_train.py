import torch
import numpy as np
from docs.VLA.Mircomanipulation_tool.model.utils import set_seed, load_data, compute_dict_mean, detach_dict
from docs.VLA.Mircomanipulation_tool.model.constants import TASK_CONFIGS
import argparse
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import yaml
from docs.VLA.Mircomanipulation_tool.model.policy import ACTPolicy
from tqdm import tqdm

def main(args):
    print("=== Parsed args ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    set_seed(args.seed)
    # Training configuration.
    ckpt_dir = args.ckpt_dir
    batch_size_train = args.batch_size
    batch_size_val = args.batch_size
    num_epochs = args.num_epochs
    chunk_size = args.chunk_size
    dataset_dir = args.dataset_dir

    task_config = TASK_CONFIGS
    # dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']

    # ACT policy configuration.
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'

    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': args.lr,
                     'num_queries': args.chunk_size,
                     'kl_weight': args.kl_weight,
                     'hidden_dim': args.hidden_dim,
                     'dim_feedforward': args.dim_feedforward,
                     'lr_backbone': lr_backbone,
                     'backbone': backbone,
                     'enc_layers': enc_layers,
                     'dec_layers': dec_layers,
                     'nheads': nheads,
                     'camera_names': camera_names,
                     }
    # Runtime training configuration.
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args.lr,
        'policy_config': policy_config,
        'seed': args.seed,
        'camera_names': camera_names,
    }

    # Load the dataset and create training and validation splits.
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size)
    # Save normalization statistics used during inference.
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Save the command-line configuration with the checkpoint.
    config_save_path = os.path.join(ckpt_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f)

    print(f'Config saved to {config_save_path}')

    # Train and save the best policy.
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, config):
    """Train the behavior-cloning policy and return its best checkpoint."""
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']
    set_seed(seed)
    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    # Track per-batch training values and per-epoch validation values.
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # Run validation before training in each epoch.
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # Validation does not require gradients.
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            # Evaluate every validation batch.
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            # Keep the checkpoint with the lowest validation loss.
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Training mode: forward pass, backpropagation, and optimizer step.
        policy.train()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Save periodic checkpoints and learning curves.
        if epoch % 2500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # Save the final and best checkpoints.
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Save final learning curves.
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """Plot training and validation histories."""
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

if __name__ == '__main__':
    task = 'Splicing_2'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dataset_dir', action='store', type=str, default=f'/home/nova/mir/dataset/dataset_{task}', help='dataset_dir')
    parser.add_argument('--ckpt_dir', action='store', type=str, default=f'/home/nova/mir/result/{task}/cs30_1e-04_161', help='ckpt_dir')
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, default=64, help='batch_size', required=False)
    parser.add_argument('--seed', action='store', type=int, default=1, help='seed', required=False)
    parser.add_argument('--num_epochs', action='store', type=int, default=5000, help='num_epochs', required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-4, required=False)

    parser.add_argument('--kl_weight', action='store', type=int, default=10, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, default=30, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=800, help='dim_feedforward', required=False)
    # parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--config', type=str, help='Path to YAML config')
    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"[WARNING] Unknown config key: {key}")
    main(args)
    # main(vars(parser.parse_args()))
