import os
import pickle

# 根目录（你可以根据需要更改）
root_dir = '/home/nova/mir'

# 遍历 epoch_0 到 epoch_49
for i in range(50):
    epoch_dir = os.path.join(root_dir, f'task_003/epoch_{i}', 'Action')
    
    if not os.path.exists(epoch_dir):
        print(f"目录不存在：{epoch_dir}")
        continue

    # 遍历该目录下的所有 pkl 文件
    for fname in os.listdir(epoch_dir):
        if not fname.endswith('.pkl'):
            continue
        
        fpath = os.path.join(epoch_dir, fname)
        index = int(fname.replace('.pkl', ''))

        # 读取 pkl 文件
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        # 检查数据格式
        if not isinstance(data, dict) or 'position' not in data:
            print(f"文件格式异常：{fpath}")
            continue

        # 修改逻辑
        if 0 <= index <= 9:
            data['position'] += 200
        elif 10 <= index <= 19:
            data['position'] -= 200
        elif 20 <= index <= 29:
            data['position'] += 100
        elif 30 <= index <= 39:
            data['position'] -= 100
        else:
            # 超出范围的不处理
            continue

        # 保存修改后的数据
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)

        print(f"已处理: {fpath}")

print("✅ 所有数据处理完成。")