import pickle
import glob
import os
import numpy as np  # 引入numpy以防数据是numpy数组格式

def modify_z_axis(file_path):
    """
    读取pkl文件，将[x, y, z]格式数据的z轴修改为0，并保存。
    """
    try:
        # 1. 读取数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        is_modified = False

        # 2. 检查并修改数据 (假设数据是 list 或 numpy array)
        # 情况 A: 数据是列表 [x, y, z]
        if isinstance(data, list) and len(data) >= 3:
            # 只有当 z 不为 0 时才修改，避免重复写入
            if data[2] != 0:
                data[2] = 0
                is_modified = True
        
        # 情况 B: 数据是 Numpy 数组 (在机器人数据中很常见)
        elif isinstance(data, np.ndarray) and data.size >= 3:
             # 处理一维数组 [x, y, z]
            if data.ndim == 1:
                if data[2] != 0:
                    data[2] = 0
                    is_modified = True
            # 如果数据是二维 [[x,y,z], ...] 这种轨迹形式，请取消下面注释
            # elif data.ndim == 2 and data.shape[1] >= 3:
            #     data[:, 2] = 0
            #     is_modified = True

        # 3. 如果发生了修改，则回写文件
        if is_modified:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[已修改] {file_path}")
        else:
            # 数据格式不对 或 Z已经是0
            # print(f"[跳过] {file_path} (无需修改或格式不符)")
            pass

    except Exception as e:
        print(f"[错误] 处理文件 {file_path} 时出错: {e}")

def main():
    base_path = "/home/nova/mir/task/task_Splicing_3"
    
    # --- 路径匹配规则 ---
    
    # 规则 1: 匹配所有 epoch_{i} 下的 Action
    # 使用通配符 * 匹配任意 epoch 编号
    action_pattern = os.path.join(base_path, "epoch_*/Action/*.pkl")
    
    # 规则 2: 仅匹配 epoch_0 下的 Observations/qpos
    # (如果你希望匹配所有 epoch 下的 qpos，请将 'epoch_0' 改为 'epoch_*')
    qpos_pattern = os.path.join(base_path, "epoch_*/Observations/qpos/*.pkl")
    
    # 获取所有符合的文件列表
    files_to_process = []
    files_to_process.extend(glob.glob(action_pattern))
    files_to_process.extend(glob.glob(qpos_pattern))
    
    print(f"共找到 {len(files_to_process)} 个文件，准备处理...")
    
    # --- 开始批量处理 ---
    for file_path in files_to_process:
        modify_z_axis(file_path)

    print("处理完成。")

if __name__ == "__main__":
    main()