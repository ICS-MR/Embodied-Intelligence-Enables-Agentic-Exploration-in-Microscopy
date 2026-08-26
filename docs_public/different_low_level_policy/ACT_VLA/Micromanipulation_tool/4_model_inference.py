import time
from einops import rearrange
import torch
import numpy as np
from model.utils import set_seed
from model.constants import TASK_CONFIGS
import argparse
import os
import pickle
from utils.task_interfaces import create_task_agent
from model.policy import ACTPolicy
import cv2
import logging

def main(args, agent):
    set_seed(1)
    ckpt_dir = args['ckpt_dir']
    task_config = TASK_CONFIGS
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    state_dim = 14
    policy_config = {'lr': args['lr'],
                     'num_queries': args['chunk_size'],
                     'kl_weight': args['kl_weight'],
                     'hidden_dim': args['hidden_dim'],
                     'dim_feedforward': args['dim_feedforward'],
                     'lr_backbone': 1e-5,
                     'backbone': 'resnet18',
                     'enc_layers': 4,
                     'dec_layers': 7,
                     'nheads': 8,
                     'camera_names': camera_names,
                     }

    config = {
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_config': policy_config,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'video_filename': args['video_filename']
    }

    eval_bc(config, 'policy_best.ckpt', agent)
    print()


def get_image(img, camera_names):
    if img is None:
        raise ValueError("Empty image; check camera connection")
    curr_images = []
    obs = {'top' : img}
    for img in camera_names:
        curr_image = rearrange(obs[img], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, agent):
    video_writer = None
    fps = 5
    video_filename = config['video_filename']

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    camera_names = config['camera_names']
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(policy_config)
    
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
    print('Policy initialized.')

    logging.info('Initialization finished')
    action_offsets = None
    with torch.inference_mode():
        for t in range(max_timesteps):
            time.sleep(0.01)
            current_pos = agent.get_ee_pos()
            if current_pos is None:
                print("Position read failed, skipping frame")
                continue

            qpos_vec = agent.get_qpos_vec()
            qpos = pre_process(qpos_vec)
            qpos_tensor = torch.from_numpy(qpos).float().cuda().view(1, -1)
            
            curr_image = agent.get_img()
            if curr_image is None:
                print("Camera read failed, skipping frame")
                continue

            if video_writer is None:
                video_dir = os.path.dirname(video_filename)
                if video_dir:
                    os.makedirs(video_dir, exist_ok=True)
                frame_size = (curr_image.shape[1], curr_image.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
            
            video_writer.write(cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR))
            curr_image = get_image(curr_image,camera_names)

            if t % query_frequency == 0:
                policy_output = policy(qpos_tensor, curr_image)
                all_actions = policy_output[0] if isinstance(policy_output, tuple) else policy_output

            if temporal_agg:
                all_time_actions[[t], t:t+num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]

            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target, action_offsets = agent.execute_action(action, current_qpos=current_pos, offsets=action_offsets)
            logging.info(f'step_{t}, current={current_pos}, target={target}')
        
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_filename}")

def start_log(task_name):
    log_dir = f'/home/nova/videos/{task_name}/{task_name}_{record_epoch}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"record.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

if __name__ == '__main__':
    task_name = 'Splicing_2'
    record_epoch = '09'
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, default=task_name, help='task_name')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'robot', 'microscope'], help='hardware backend')
    parser.add_argument('--control_mode', type=str, default='auto', choices=['auto', 'xy', 'z', 'brightness', 'exposure'], help='controlled state')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='robot serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='robot serial baudrate')
    parser.add_argument('--timeout', type=float, default=0.1, help='I/O timeout')
    parser.add_argument('--record_epoch', action='store', type=str, default=record_epoch, help='record_epoch')
    parser.add_argument('--video_filename', action='store', type=str, default=None, help='video_filename')
    parser.add_argument('--ckpt_dir', action='store', type=str, default=None, help='ckpt_dir')
    parser.add_argument('--batch_size', action='store', type=int, default=64, help='batch_size', required=False)
    parser.add_argument('--seed', action='store', type=int, default=1, help='seed', required=False)
    parser.add_argument('--num_epochs', action='store', type=int, default=5000, help='num_epochs', required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-04, required=False)
    parser.add_argument('--kl_weight', action='store', type=int, default=10, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, default=30, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=800, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', default=False, action='store_true')
    args = vars(parser.parse_args())

    task_name = args['task_name']
    record_epoch = args['record_epoch']
    if args['video_filename'] is None:
        args['video_filename'] = f'/home/nova/videos/{task_name}/{task_name}_{record_epoch}/video.avi'
    if args['ckpt_dir'] is None:
        args['ckpt_dir'] = f'/home/nova/mir/result/{task_name}/cs30_1e-04'

    agent = create_task_agent(
        task_name,
        port_id=args['port'],
        baudrate=args['baudrate'],
        timeout=args['timeout'],
        backend=args['backend'],
        control_mode=args['control_mode'],
    )
    try:
        start_log(task_name)
        agent.open()
        main(args, agent)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
    finally:
        agent.close()
