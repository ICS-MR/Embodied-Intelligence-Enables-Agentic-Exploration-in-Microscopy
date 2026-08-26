import torch
import numpy as np
from model.utils import set_seed, load_data, compute_dict_mean, detach_dict
from model.constants import TASK_CONFIGS
import argparse
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import yaml
from model.policy import ACTPolicy
from tqdm import tqdm

def main(args):
    print("=== Parsed args ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    set_seed(args.seed)
    #****************************************** Training parameters ************************************************************
    ckpt_dir = args.ckpt_dir                 # Directory for saving checkpoints
    batch_size_train = args.batch_size       # Training batch size
    batch_size_val = args.batch_size         # Validation batch size
    num_epochs = args.num_epochs             # Number of training epochs
    chunk_size = args.chunk_size             # Chunk size
    dataset_dir = args.dataset_dir           # Dataset directory containing episodes

    task_config = TASK_CONFIGS
    # dataset_dir = task_config['dataset_dir']    # Dataset directory containing episodes
    num_episodes = task_config['num_episodes']  # Number of samples
    camera_names = task_config['camera_names']  # Camera names

    #****************************************** ACT policy parameters ************************************************************
    state_dim = 14                              # State dimension
    lr_backbone = 1e-5                          # Backbone learning rate
    backbone = 'resnet18'                       # Backbone used for image preprocessing and feature extraction

    enc_layers = 4                              # Number of encoder layers
    dec_layers = 7                              # Number of decoder layers
    nheads = 8                                  # Number of heads in multi-head attention
    # ACTPolicy configuration parameters
    policy_config = {'lr': args.lr,                              # Learning rate
                     'num_queries': args.chunk_size,             # Chunk size, approximately the number of actions predicted per step
                     'kl_weight': args.kl_weight,                # KL divergence weight
                     'hidden_dim': args.hidden_dim,              # Hidden dimension
                     'dim_feedforward': args.dim_feedforward,    # Feedforward network dimension
                     'lr_backbone': lr_backbone,                    # Backbone learning rate
                     'backbone': backbone,                          # Backbone for image preprocessing and feature extraction
                     'enc_layers': enc_layers,                      # Number of encoder layers
                     'dec_layers': dec_layers,                      # Number of decoder layers
                     'nheads': nheads,                              # Number of multi-head attention heads
                     'camera_names': camera_names,                  # Cameras
                     }
    # Configure training parameters.
    config = {
        'num_epochs': num_epochs,               # Number of training epochs
        'ckpt_dir': ckpt_dir,                   # Model output path
        'state_dim': state_dim,                 # State dimension; not used during training
        'lr': args.lr,                       # Learning rate
        'policy_config': policy_config,         # Policy configuration
        'seed': args.seed,                   # Random seed
        'camera_names': camera_names,           # Camera names
        # 'episode_len': episode_len,           # Simulation horizon
    }

    # Load data and split it into training and validation sets.
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size)
    # Save dataset statistics.
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Save the configuration used for this run in the model output directory.
    config_save_path = os.path.join(ckpt_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f)  # Write the current parameters in YAML format.

    print(f'Config saved to {config_save_path}')

    # Train the model.
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # Save the best model.
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

# Forward-pass function used to calculate the loss.
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)  # Forward pass

def train_bc(train_dataloader, val_dataloader, config):
    """
    Behavior-cloning training function.
    :param train_dataloader: Training set
    :param val_dataloader: Validation set
    :param config: Configuration for right_target_angles
    :return: Best model
    """
    # Load configuration parameters.
    num_epochs = config['num_epochs']           # Number of training epochs
    ckpt_dir = config['ckpt_dir']               # Dataset path
    seed = config['seed']                       # Random seed
    policy_config = config['policy_config']     # Policy configuration
    set_seed(seed)
    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    # Initialize training and validation history.
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # Start the loop; tqdm adds a progress bar for training progress.
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        '''
        Validation mode: disable gradients, run forward inference, calculate loss,
        and save the loss and related information to the validation history.
        '''
        # Disable gradient operations to avoid unnecessary computation and memory usage.
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            # Iterate over the validation set, run forward inference, and calculate loss.
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            # Extract and update the minimum validation loss and related information.
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        '''
        Training mode: calculate gradients, run forward and backward passes,
        update parameters, and save results.
        '''
        policy.train()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # Run backpropagation and update parameters.
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

        # Save the model and plot training curves every 1,000 epochs.
        if epoch % 2500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # Save the final weights after training.
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Plot training curves.
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """
    Plot training curves.
    """
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
    parser.add_argument('--dataset_dir', action='store', type=str, default=f'/home/nova/mir/dataset/dataset_{task}', help='dataset_dir')   # Dataset path; update for each run
    parser.add_argument('--ckpt_dir', action='store', type=str, default=f'/home/nova/mir/result/{task}/cs30_1e-04_161', help='ckpt_dir')      # Model output path; update for each run
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, default= 64, help='batch_size', required=False)               # Larger batches may improve training but consume more GPU memory
    parser.add_argument('--seed', action='store', type=int, default= 1, help='seed', required=False)                            # Random seed for reproducible sequences and training results
    parser.add_argument('--num_epochs', action='store', type=int, default= 5000, help='num_epochs', required=False)             # More epochs may improve results but increase training time; typically 5,000-8,000
    parser.add_argument('--lr', action='store', type=float, help='lr',default= 1e-4, required=False)                            # Tune the learning rate; high values destabilize training and low values slow convergence

    parser.add_argument('--kl_weight', action='store', type=int, default= 10 ,help='KL Weight', required=False)                  # KL-divergence weight; tune because it affects results
    parser.add_argument('--chunk_size', action='store', type=int, default= 30 ,help='chunk_size', required=False)                # Approximate number of actions predicted per step; affects results
    parser.add_argument('--hidden_dim', action='store', type=int, default= 512 ,help='hidden_dim', required=False)               # Hidden dimension; affects results
    parser.add_argument('--dim_feedforward', action='store', type=int, default= 800 ,help='dim_feedforward', required=False)     # Feedforward network dimension; affects results
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
