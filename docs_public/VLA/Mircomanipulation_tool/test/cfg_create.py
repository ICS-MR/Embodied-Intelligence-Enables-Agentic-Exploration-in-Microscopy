import os

# Base configuration template.
config_template = """# Use a separate checkpoint directory for each experiment.
ckpt_dir: "./data/result/{task}/cs{chunk_size}_{lr_str}"
dataset_dir: "./data/dataset/dataset_{task}"

# Training hyperparameters
batch_size: 64          # Batch size
seed: 1                 # Random seed
num_epochs: 5000        # Training epochs
lr: {lr}                # Learning rate

# Model architecture and loss
kl_weight: 10           # KL-divergence loss weight
chunk_size: {chunk_size}  # Action chunk size
hidden_dim: 512         # Transformer hidden dimension
dim_feedforward: 800    # Feed-forward dimension
"""

# Parameter combinations.
chunk_sizes = [10, 15, 20, 25, 30, 50]
learning_rates = [1e-4]
task = 'Splicing_2'
# Generate every configuration.
output_dir = "configs"
os.makedirs(output_dir, exist_ok=True)

for cs in chunk_sizes:
    for lr in learning_rates:
        # Preserve compact scientific notation such as 1e-3.
        lr_str = f"{lr:.1e}".replace('.0', '').replace('+', '')
        
        # Build a descriptive filename.
        filename = f"{task}_cs{cs}_{lr_str}.yaml"
        
        # Fill template parameters.
        config = config_template.format(
            chunk_size=cs,
            lr=lr,
            lr_str=lr_str,
            task=task
        )
        
        # Save the generated configuration.
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(config)
        print(f"Generated: {filename}")

print(f"\nGenerated {len(chunk_sizes) * len(learning_rates)} configuration files")
print(f"Output directory: {os.path.abspath(output_dir)}")
