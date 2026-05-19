import os
import pickle
import matplotlib.pyplot as plt

def plot_xy_trajectory(action_folder, i):
    """可视化XY平面运动轨迹"""
    # 初始化坐标容器
    x_values = []
    y_values = []
    
    # 获取排序后的pkl文件列表
    pkl_files = sorted(
        [f for f in os.listdir(action_folder) if f.endswith('.pkl')],
        key=lambda x: int(x.split('.')[0])  # 按数字顺序排序
    )

    # 读取数据并提取XY坐标
    for file in pkl_files:
        with open(os.path.join(action_folder, file), 'rb') as f:
            # 假设坐标格式为 [x, y, z] 的三维坐标
            coord = pickle.load(f)
            x_values.append(coord[0])  # 取第一个元素为X坐标
            y_values.append(coord[1])  # 取第二个元素为Y坐标

    # 创建可视化图形
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'g->', linewidth=1.5, markersize=8, label='运动轨迹')
    plt.scatter(x_values[0], y_values[0], c='blue', s=100, label='起点')
    plt.scatter(x_values[-1], y_values[-1], c='red', s=100, label='终点')
    
    # 添加图形装饰
    plt.title(f"FIG_{i}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 自动调整坐标范围
    plt.xlim(min(x_values)-500, max(x_values)+500)
    plt.ylim(min(y_values)-500, max(y_values)+500)
    
    plt.show()

# 使用示例（路径需要替换为实际Action文件夹路径）
for i in range(0, 63):
    plot_xy_trajectory(f'/home/nova/mir/task/task_Splicing_3/epoch_{i}/Action', i)
# plot_xy_trajectory('/home/nova/mir/task_111/epochs/epoch_0/Action', i = 0)