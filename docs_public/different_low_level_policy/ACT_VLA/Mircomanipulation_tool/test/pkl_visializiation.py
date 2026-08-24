import os
import pickle
import matplotlib.pyplot as plt

def plot_xy_trajectory(action_folder, i):
    """Visualize a motion trajectory in the XY plane."""
    # Initialize coordinate containers.
    x_values = []
    y_values = []
    
    # Get the sorted list of PKL files.
    pkl_files = sorted(
        [f for f in os.listdir(action_folder) if f.endswith('.pkl')],
        key=lambda x: int(x.split('.')[0])  # Sort numerically.
    )

    # Read the data and extract XY coordinates.
    for file in pkl_files:
        with open(os.path.join(action_folder, file), 'rb') as f:
            # Assume three-dimensional coordinates in [x, y, z] format.
            coord = pickle.load(f)
            x_values.append(coord[0])  # Use the first element as the X coordinate.
            y_values.append(coord[1])  # Use the second element as the Y coordinate.

    # Create the visualization.
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'g->', linewidth=1.5, markersize=8, label='Motion Trajectory')
    plt.scatter(x_values[0], y_values[0], c='blue', s=100, label='Start')
    plt.scatter(x_values[-1], y_values[-1], c='red', s=100, label='End')
    
    # Add plot decorations.
    plt.title(f"FIG_{i}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust coordinate limits automatically.
    plt.xlim(min(x_values)-500, max(x_values)+500)
    plt.ylim(min(y_values)-500, max(y_values)+500)
    
    plt.show()

# Usage example; replace the path with the actual Action folder.
for i in range(0, 63):
    plot_xy_trajectory(f'/home/nova/mir/task/task_Splicing_3/epoch_{i}/Action', i)
# plot_xy_trajectory('/home/nova/mir/task_111/epochs/epoch_0/Action', i = 0)
