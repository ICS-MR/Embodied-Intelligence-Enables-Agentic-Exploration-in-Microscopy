import os
import pickle
import numpy as np

# 基础路径
base_path = "/home/nova/mir/task/task_Splicing_3"

# 遍历每个 epoch 文件夹
for epoch_dir in sorted(os.listdir(base_path)):
    if not epoch_dir.startswith("epoch_"):
        continue
    
    epoch_path = os.path.join(base_path, epoch_dir, "Action")
    if not os.path.exists(epoch_path):
        print(f"警告: {epoch_path} 不存在，跳过")
        continue
    
    # 收集所有 pkl 文件的均值
    epoch_means = []
    
    # 遍历 Action 目录下的所有 pkl 文件
    for pkl_file in sorted(os.listdir(epoch_path)):
        if not pkl_file.endswith(".pkl"):
            continue
        
        file_path = os.path.join(epoch_path, pkl_file)
        
        # 加载 pkl 文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 确保数据是 [x, y, z] 格式
        if not isinstance(data, (list, np.ndarray)) or len(data) != 3:
            print(f"警告: {file_path} 数据格式不符合 [x,y,z]，跳过")
            continue
        
        # 计算均值
        mean_values = np.mean(data, axis=0) if isinstance(data, np.ndarray) else np.mean(data)
        epoch_means.append(mean_values)
    
    # 计算整个 epoch 的均值
    if epoch_means:
        overall_mean = np.mean(epoch_means, axis=0)
        # if overall_mean > 10000 or overall_mean < -10000:
        #     print(f"{epoch_dir}: 均值 = {overall_mean}")
        print(f"{epoch_dir}: 均值 = {overall_mean}")
    else:
        print(f"{epoch_dir}: 无有效数据")