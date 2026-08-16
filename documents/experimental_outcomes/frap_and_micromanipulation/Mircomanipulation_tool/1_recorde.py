import time
import pickle
import os
import cv2
import shutil
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from docs.VLA.Mircomanipulation_tool.utils.task_interfaces import create_task_agent


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_next_epoch(root_folder):
    if not os.path.exists(root_folder):
        return 0
    existing_epochs = [
        int(name.split('_')[-1]) for name in os.listdir(root_folder) if name.startswith("epoch_")
    ]
    return max(existing_epochs) + 1 if existing_epochs else 0

def save_step(img, qpos, stage_num, save_count, action_folder, image_folder, qpos_folder, stage_folder):
    if img is not None:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(image_folder, f'img_{save_count}.png'), img_bgr)

    with open(os.path.join(action_folder, f'{save_count}.pkl'), 'wb') as f:
        pickle.dump(qpos, f)

    with open(os.path.join(qpos_folder, f'{save_count}.pkl'), 'wb') as f:
        pickle.dump(qpos, f)

    with open(os.path.join(stage_folder, f'{save_count}.pkl'), 'wb') as f:
        pickle.dump([stage_num], f)
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='task_Splicing_3', help='task name')
    parser.add_argument('--root_folder', type=str, default='/home/nova/mir/task', help='dataset root')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'robot', 'microscope'], help='hardware backend')
    parser.add_argument('--control_mode', type=str, default='auto', choices=['auto', 'xy', 'z', 'brightness', 'exposure'], help='controlled state')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='robot serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='robot serial baudrate')
    parser.add_argument('--timeout', type=float, default=0.1, help='I/O timeout')
    args = parser.parse_args()

    task_name = args.task_name
    root_folder = args.root_folder
    agent = create_task_agent(
        task_name,
        port_id=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        backend=args.backend,
        control_mode=args.control_mode,
    )

    root_folder = os.path.join(root_folder, task_name)
    
    agent.open()

    try:
        while True:
            if agent.delete_Flag:
                latest_epoch = get_next_epoch(root_folder) - 1
                delete_folder = os.path.join(root_folder, f'epoch_{latest_epoch}')
                if os.path.exists(delete_folder):
                    print(f"Deleting latest epoch folder: {delete_folder}")
                    shutil.rmtree(delete_folder)
                else:
                    print(f"No epoch folder to delete: {delete_folder}")
                agent.delete_Flag = False
                continue

            if agent.should_exit:
                time.sleep(1)
                break

            if agent.recording_Flag:
                epoch = get_next_epoch(root_folder)
                epoch_folder = os.path.join(root_folder, f'epoch_{epoch}')

                action_folder = os.path.join(epoch_folder, 'Action')
                image_folder = os.path.join(epoch_folder, 'Observations', 'img')
                qpos_folder = os.path.join(epoch_folder, 'Observations', 'qpos')
                stage_folder = os.path.join(epoch_folder, 'Observations', 'stage')

                for folder in [action_folder, image_folder, qpos_folder, stage_folder]:
                    create_folder(folder)

                print(f'Starting epoch {epoch}, saving to: {epoch_folder}')
                print(f'Start epoch {epoch}')

                save_count = 0
                interval = agent.profile.interval
                next_time = time.perf_counter()
                save_pool = ThreadPoolExecutor(max_workers=4)
                futures = []

                try:
                    while agent.recording_Flag:
                        img = agent.get_img()
                        qpos = agent.get_ee_pos()
                        if img is None or qpos is None:
                            print("[WARN] No synchronized snapshot, skipping frame")
                            next_time += interval
                            time.sleep(max(0.0, next_time - time.perf_counter()))
                            continue

                        stage_num = agent.get_current_stage()
                        future = save_pool.submit(
                            save_step,
                            img,
                            qpos,
                            stage_num,
                            save_count,
                            action_folder,
                            image_folder,
                            qpos_folder,
                            stage_folder,
                        )
                        futures.append(future)

                        if len(futures) > 128:
                            futures = [item for item in futures if not item.done()]

                        print(f"step_{save_count}, qpos={np.round(qpos, 2)}, stage={stage_num}")

                        save_count += 1
                        next_time += interval
                        time.sleep(max(0.0, next_time - time.perf_counter()))
                finally:
                    print("Waiting for pending save tasks...")
                    for future in futures:
                        future.result()
                    save_pool.shutdown(wait=True)
            else:
                time.sleep(0.01)
    finally:
        agent.close()
