import os
import pickle
import pprint

# ✅ 设置你的目标文件夹路径
folder_path = "/home/nova/mir/task/task_Splicing_3/epoch_0/Observations/stage"

def load_and_print_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("内容:")
        pprint.pprint(data, depth=5, compact=True)
    except Exception as e:
        print(f"❌ 加载失败: {file_path}, 错误信息: {e}")

def main():
    if not os.path.isdir(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    for i in range(600):
        file_path = os.path.join(folder_path, f"{i}.pkl")
        if os.path.isfile(file_path):
            load_and_print_pkl(file_path)
if __name__ == "__main__":
    main()
