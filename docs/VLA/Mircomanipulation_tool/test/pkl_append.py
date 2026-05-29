import os
import shutil
import pickle

def get_max_index(files, ext):
    """从文件名中提取最大编号"""
    indexes = []
    for f in files:
        if f.endswith(ext):
            try:
                indexes.append(int(f.replace(ext, '')))
            except ValueError:
                continue
    return max(indexes) if indexes else -1

# 设置根路径
root_dir = '/home/nova/mir/task_003_2'

# 遍历 50 个 epoch 文件夹
for i in range(50):
    print(f'▶ 正在处理 epoch_{i}...')
    epoch_path = os.path.join(root_dir, f'epoch_{i}')
    action_path = os.path.join(epoch_path, 'Action')
    img_path = os.path.join(epoch_path, 'Observations', 'img')

    # 获取最大编号
    action_files = os.listdir(action_path)
    img_files = os.listdir(img_path)

    max_action_idx = get_max_index(action_files, '.pkl')
    max_img_idx = max_action_idx

    # 构造源文件路径
    action_src = os.path.join(action_path, f'{max_action_idx}.pkl')
    img_src = os.path.join(img_path, f'img_{max_img_idx}.png')

    # 扩增 Action
    with open(action_src, 'rb') as f:
        action_data = pickle.load(f)

    for j in range(1, 4):
        new_idx = max_action_idx + j
        new_path = os.path.join(action_path, f'{new_idx}.pkl')
        with open(new_path, 'wb') as f:
            pickle.dump(action_data, f)
        # print(f'  ✓ 生成 Action/{new_idx}.pkl')

    # 扩增图像
    for j in range(1, 4):
        new_idx = max_img_idx + j
        new_path = os.path.join(img_path, f'img_{new_idx}.png')
        shutil.copy(img_src, new_path)
        # print(f'  ✓ 复制 img/img_{new_idx}.png')

print('\n✅ 所有 epoch 数据扩增完毕！')
