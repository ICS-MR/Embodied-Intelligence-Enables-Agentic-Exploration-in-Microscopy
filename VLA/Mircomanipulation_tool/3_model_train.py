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
    #****************************************** 训练参数 ************************************************************
    ckpt_dir = args.ckpt_dir                 # 保存checkpoint的目录
    batch_size_train = args.batch_size       # 训练batch大小
    batch_size_val = args.batch_size         # 验证batch大小
    num_epochs = args.num_epochs             # 训练周期次数
    chunk_size = args.chunk_size             # 分块大小
    dataset_dir = args.dataset_dir           # 数据集地址，保存epoch的地址

    task_config = TASK_CONFIGS
    # dataset_dir = task_config['dataset_dir']    # 数据集地址，保存epoch的地址
    num_episodes = task_config['num_episodes']  # 样本数
    camera_names = task_config['camera_names']  # 相机名称

    #****************************************** ACT策略参数 ************************************************************
    state_dim = 14                              # 状态维度
    lr_backbone = 1e-5                          # 主干网络学习率
    backbone = 'resnet18'                       # 主干网络，一般用来进行图片预处理，提取特征图

    enc_layers = 4                              # 编码器层数
    dec_layers = 7                              # 解码器层数
    nheads = 8                                  # 注意力机制头数，因为采用多头注意力机制
    # ACTPolicy配置参数
    policy_config = {'lr': args.lr,                              # 学习率
                     'num_queries': args.chunk_size,             # 分块大小，大致相当于一步预测几个动作
                     'kl_weight': args.kl_weight,                # KL散度
                     'hidden_dim': args.hidden_dim,              # 隐层维度
                     'dim_feedforward': args.dim_feedforward,    # 前馈网络维度
                     'lr_backbone': lr_backbone,                    # 主干网络学习率
                     'backbone': backbone,                          # 主干网络，一般用来进行图片预处理，提取特征图
                     'enc_layers': enc_layers,                      # 编码器层数
                     'dec_layers': dec_layers,                      # 解码器层数
                     'nheads': nheads,                              # 注意力机制头数，因为采用多头注意力机制
                     'camera_names': camera_names,                  # 相机
                     }
    # 配置训练参数
    config = {
        'num_epochs': num_epochs,               # 训练轮次
        'ckpt_dir': ckpt_dir,                   # 模型保存路径
        'state_dim': state_dim,                 # 状态维度 ， 在训练时没有用到
        'lr': args.lr,                       # 学习率
        'policy_config': policy_config,         # 策略配置
        'seed': args.seed,                   # 随机种子
        'camera_names': camera_names,           # 相机名称
        # 'episode_len': episode_len,           # 仿真步长
    }

    # 加载数据，将数据分为训练集和验证集
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size)
    # 保存数据统计信息
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 保存本次使用的配置参数到模型保存目录
    config_save_path = os.path.join(ckpt_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f)  # 把当前参数写入yaml格式

    print(f'Config saved to {config_save_path}')

    # 训练模型
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # 保存最优模型
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

# 前向传播函数，用于进行前向传播，计算损失
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)  # 前向传播

def train_bc(train_dataloader, val_dataloader, config):
    """
    BC训练函数
    :param train_dataloader: 训练集
    :param val_dataloader: 验证集
    :param config: 配置right_target_angles
    :return: 最优模型
    """
    # 加载配置参数
    num_epochs = config['num_epochs']           # 训练轮次
    ckpt_dir = config['ckpt_dir']               # 数据集地址
    seed = config['seed']                       # 随机种子
    policy_config = config['policy_config']     # 策略配置
    set_seed(seed)
    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    # 初始化训练和验证历史记录
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    #开启循环，tqdm用于增加进度条，显示训练进度
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        '''
        验证模式：禁用梯度计算，前向传播进行推理，并计算出损失, 并将损失和相关信息保存到验证历史记录中。
        '''
        # 禁用所有与梯度计算相关的操作，避免不必要的计算和内存占用。
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            # 对验证集进行遍历，前向传播进行推理，并计算出损失
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            # 提取并更新最低验证损失及其相关信息
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
        训练模式：进行梯度计算，前向传播，反向传播，更新参数，并保存。
        '''
        policy.train()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # 进行反向传播，更新参数
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

        # 每1000次进行保存模型， 并进行绘制训练曲线图
        if epoch % 2500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # 结束后保存最后一次权重
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # 绘制训练曲线图
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """
    绘制训练曲线图
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
    parser.add_argument('--dataset_dir', action='store', type=str, default=f'/home/nova/mir/dataset/dataset_{task}', help='dataset_dir')   # 数据集路径，每次更改
    parser.add_argument('--ckpt_dir', action='store', type=str, default=f'/home/nova/mir/result/{task}/cs30_1e-04_161', help='ckpt_dir')      # 模型保存地址，每次更改
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, default= 64, help='batch_size', required=False)               # 训练batch大小，越大对训练越有效，但是对显存占用也更大
    parser.add_argument('--seed', action='store', type=int, default= 1, help='seed', required=False)                            # 随机种子，用于生成随机序列，保证每次训练结果一致，对结果有影响
    parser.add_argument('--num_epochs', action='store', type=int, default= 5000, help='num_epochs', required=False)             # 训练次数，一般轮次越多越好，但是训练时间也越长，5000-8000次
    parser.add_argument('--lr', action='store', type=float, help='lr',default= 1e-4, required=False)                            # 学习率，需要微调，过大会导致训练不稳定，过小会导致收敛速度太慢

    parser.add_argument('--kl_weight', action='store', type=int, default= 10 ,help='KL Weight', required=False)                  # KL散度，需要微调，对结果有影响
    parser.add_argument('--chunk_size', action='store', type=int, default= 30 ,help='chunk_size', required=False)                # 分块大小，大致相当于一步预测几个动作，对结果有影响
    parser.add_argument('--hidden_dim', action='store', type=int, default= 512 ,help='hidden_dim', required=False)               # 隐层维度，对结果有影响
    parser.add_argument('--dim_feedforward', action='store', type=int, default= 800 ,help='dim_feedforward', required=False)     # 前馈网络维度，对结果有影响
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