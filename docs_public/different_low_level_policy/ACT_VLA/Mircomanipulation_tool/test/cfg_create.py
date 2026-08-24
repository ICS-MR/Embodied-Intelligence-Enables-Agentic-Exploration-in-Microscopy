import os

# Base configuration template
config_template = """# Model output path; use a separate directory for each experiment
ckpt_dir: "/home/nova/mir/result/{task}/cs{chunk_size}_{lr_str}"
dataset_dir: "/home/nova/mir/dataset/dataset_{task}"

# Training hyperparameters
batch_size: 64          # Batch size
seed: 1                 # Random seed
num_epochs: 5000        # Number of training epochs
lr: {lr}                # Learning rate

# Model architecture and loss settings
kl_weight: 10            # Weight of the KL-divergence loss
chunk_size: {chunk_size}           # Action chunk size
hidden_dim: 512         # Transformer hidden dimension
dim_feedforward: 800    # Feedforward layer dimension
"""

# Parameter combinations
chunk_sizes = [10, 15, 20, 25, 30, 50]
learning_rates = [1e-4]
task = 'Splicing_2'
# Generate all configurations.
output_dir = "configs"
os.makedirs(output_dir, exist_ok=True)

for cs in chunk_sizes:
    for lr in learning_rates:
        # Preserve scientific notation, such as 1e-3 or 5e-4.
        lr_str = f"{lr:.1e}".replace('.0', '').replace('+', '')
        
        # Generate the filename in the requested format.
        filename = f"{task}_cs{cs}_{lr_str}.yaml"
        
        # Substitute parameters into the template.
        config = config_template.format(
            chunk_size=cs,
            lr=lr,
            lr_str=lr_str,
            task=task
        )
        
        # Save the file.
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(config)
        print(f"Generated: {filename}")

print(f"\nGenerated {len(chunk_sizes)*len(learning_rates)} configuration files")
print(f"Output path: {os.path.abspath(output_dir)}")
