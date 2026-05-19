import os

# 基础配置模板
config_template = """# 模型保存路径（每个实验建议改一个目录）
ckpt_dir: "/home/nova/mir/result/{task}/cs{chunk_size}_{lr_str}"
dataset_dir: "/home/nova/mir/dataset/dataset_{task}"

# 训练超参数
batch_size: 64          # 批大小
seed: 1                 # 随机种子
num_epochs: 5000        # 训练轮数
lr: {lr}                # 学习率

# 模型结构/损失函数相关
kl_weight: 10            # KL 散度损失的权重
chunk_size: {chunk_size}           # 动作分块大小
hidden_dim: 512         # Transformer 隐藏层维度
dim_feedforward: 800    # 前馈层维度
"""

# 参数组合
chunk_sizes = [10, 15, 20, 25, 30, 50]
learning_rates = [1e-4]
task = 'Splicing_2'
# 生成所有配置
output_dir = "configs"
os.makedirs(output_dir, exist_ok=True)

for cs in chunk_sizes:
    for lr in learning_rates:
        # 保持原始科学计数法格式（如 1e-3, 5e-4）
        lr_str = f"{lr:.1e}".replace('.0', '').replace('+', '')
        
        # 生成文件名（完全按照您的要求）
        filename = f"{task}_cs{cs}_{lr_str}.yaml"
        
        # 替换模板中的参数
        config = config_template.format(
            chunk_size=cs,
            lr=lr,
            lr_str=lr_str,
            task=task
        )
        
        # 保存文件
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(config)
        print(f"生成: {filename}")

print(f"\n共生成 {len(chunk_sizes)*len(learning_rates)} 个配置文件")
print(f"保存路径: {os.path.abspath(output_dir)}")