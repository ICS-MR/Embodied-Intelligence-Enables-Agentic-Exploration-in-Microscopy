import os
import argparse

def count_files(path):
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            total += len(files)
        return total
    except FileNotFoundError:
        print(f"错误：路径不存在 - {path}")
        return 0

if __name__ == "__main__":
    base_path = '/home/nova/mir/task/task_Splicing_3_language_correction_v2'
    
    # 遍历所有 epoch 目录
    total_files = 0
    print("正在扫描训练任务文件...")
    print(f"{'Epoch':<10} | {'文件数量':>10}")
    print("-" * 25)
    
    for dir_name in os.listdir(base_path):
        if dir_name.startswith("epoch_"):
            action_path = os.path.join(base_path, dir_name, "Action")
            if os.path.exists(action_path):
                count = count_files(action_path)
                print(f"{dir_name:<10} | {count:>10,}")
                total_files += count
                
    print("-" * 25)
    print(f"总文件数量: {total_files:,}")