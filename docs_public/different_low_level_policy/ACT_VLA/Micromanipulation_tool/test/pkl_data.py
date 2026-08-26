import os
import pickle
import pprint

# ✅ Set the target folder path.
folder_path = "/home/nova/mir/task/task_Splicing_3/epoch_0/Observations/stage"

def load_and_print_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Contents:")
        pprint.pprint(data, depth=5, compact=True)
    except Exception as e:
        print(f"❌ Failed to load {file_path}. Error: {e}")

def main():
    if not os.path.isdir(folder_path):
        print(f"❌ Folder does not exist: {folder_path}")
        return

    for i in range(600):
        file_path = os.path.join(folder_path, f"{i}.pkl")
        if os.path.isfile(file_path):
            load_and_print_pkl(file_path)
if __name__ == "__main__":
    main()
