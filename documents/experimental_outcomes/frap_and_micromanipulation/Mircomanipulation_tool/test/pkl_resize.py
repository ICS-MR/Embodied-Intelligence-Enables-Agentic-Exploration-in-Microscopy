import os
import pickle

# Recording root; adjust as needed.
root_dir = '/home/nova/mir'

# Process epoch_0 through epoch_49.
for i in range(50):
    epoch_dir = os.path.join(root_dir, f'task_003/epoch_{i}', 'Action')
    
    if not os.path.exists(epoch_dir):
        print(f"Directory does not exist: {epoch_dir}")
        continue

    # Process every pickle in this directory.
    for fname in os.listdir(epoch_dir):
        if not fname.endswith('.pkl'):
            continue
        
        fpath = os.path.join(epoch_dir, fname)
        index = int(fname.replace('.pkl', ''))

        # Read one pickle.
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        # Validate its structure.
        if not isinstance(data, dict) or 'position' not in data:
            print(f"Unexpected file format: {fpath}")
            continue

        # Adjust positions by index range.
        if 0 <= index <= 9:
            data['position'] += 200
        elif 10 <= index <= 19:
            data['position'] -= 200
        elif 20 <= index <= 29:
            data['position'] += 100
        elif 30 <= index <= 39:
            data['position'] -= 100
        else:
            # Ignore indices outside the configured ranges.
            continue

        # Save the modified data.
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Processed: {fpath}")

print("Finished processing all data.")
