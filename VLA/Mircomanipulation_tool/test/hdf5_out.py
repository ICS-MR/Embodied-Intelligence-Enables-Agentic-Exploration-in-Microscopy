# import h5py

# with h5py.File('/home/nova/mir/dataset/dataset_Splicing_2/episode_124.hdf5', 'r') as f:
#     actions = f['/action'][:200]  # 取前20帧
#     print("Example actions:")
#     for i, a in enumerate(actions):
#         print(f"{i}: {a}")
import h5py
import numpy as np
import os

# 设置数据集目录路径
dataset_dir = '/home/nova/mir/dataset/dataset_Splicing_2/'

# 获取目录下所有hdf5文件
episode_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
episode_files.sort()  # 按文件名排序

print(f"找到 {len(episode_files)} 个episode文件")

# 遍历每个episode文件
for episode_file in episode_files:
    file_path = os.path.join(dataset_dir, episode_file)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 检查是否存在action数据集
            if 'action' not in f:
                print(f"{episode_file}: 没有找到action数据集")
                continue
                
            actions = f['/action'][:]  # 读取所有action数据
            action_mean = np.mean(actions, axis=0)  # 计算均值
            
            print(f"{episode_file}: Action均值 = {action_mean}")
            
    except Exception as e:
        print(f"处理文件 {episode_file} 时出错: {str(e)}")