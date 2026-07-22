import os
import shutil
import pickle

def get_max_index(files, ext):
    """Extract the largest numeric index from a sequence of filenames."""
    indexes = []
    for f in files:
        if f.endswith(ext):
            try:
                indexes.append(int(f.replace(ext, '')))
            except ValueError:
                continue
    return max(indexes) if indexes else -1

# Recording root.
root_dir = '/home/nova/mir/task_003_2'

# Extend 50 epoch directories.
for i in range(50):
    print(f'Processing epoch_{i}...')
    epoch_path = os.path.join(root_dir, f'epoch_{i}')
    action_path = os.path.join(epoch_path, 'Action')
    img_path = os.path.join(epoch_path, 'Observations', 'img')

    # Find the final recorded index.
    action_files = os.listdir(action_path)
    img_files = os.listdir(img_path)

    max_action_idx = get_max_index(action_files, '.pkl')
    max_img_idx = max_action_idx

    # Build source paths for the final action and image.
    action_src = os.path.join(action_path, f'{max_action_idx}.pkl')
    img_src = os.path.join(img_path, f'img_{max_img_idx}.png')

    # Duplicate the final action three times.
    with open(action_src, 'rb') as f:
        action_data = pickle.load(f)

    for j in range(1, 4):
        new_idx = max_action_idx + j
        new_path = os.path.join(action_path, f'{new_idx}.pkl')
        with open(new_path, 'wb') as f:
            pickle.dump(action_data, f)
        # print(f'  Generated Action/{new_idx}.pkl')

    # Duplicate the final image three times.
    for j in range(1, 4):
        new_idx = max_img_idx + j
        new_path = os.path.join(img_path, f'img_{new_idx}.png')
        shutil.copy(img_src, new_path)
        # print(f'  Copied img/img_{new_idx}.png')

print('\nFinished extending all epoch data.')
