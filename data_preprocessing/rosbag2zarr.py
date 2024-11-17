import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
# from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import shutil

from scipy.spatial.transform import Rotation as R
import glob
import math

# diffusion lib======================================================
from utils.replay_buffer import ReplayBuffer
import time
from tqdm import tqdm
# diffusion======================================================


from utils.rosbag2numpy import use_rosbag_to_show, check_folder


if __name__ == "__main__":
    bag_folder_name="pick_place_241021_kcds21f"
    bag_folder_path="/media/camille/SATA/dataset/"+bag_folder_name
    
    # bag_folder_name="pick_place_2024-10-21"
    # bag_folder_path = "/home/camille/data_utils/rosbag/"+bag_folder_name
    save_zarr_folder = f"{bag_folder_path}/zarr" 
    check_folder(save_zarr_folder)  
    bagpath = sorted(glob.glob(f"{bag_folder_path}/*.bag"))
    print(bagpath)
    print(len(bagpath))
    output_zarr_path = f"{save_zarr_folder}/{bag_folder_name}_v2.zarr"
    
    replay_buffer = ReplayBuffer.create_from_path(output_zarr_path, mode='a')
    error_bag = []
    ok_bag = []
    for path in tqdm(bagpath, desc="Processing bags", unit="bag"):
        start_time = time.time()
        print("current path", path)
        seed = replay_buffer.n_episodes
        try:
            img01,img02, aligned_state_joint,aligned_cmd_joint = use_rosbag_to_show(bag_folder_path, bag_path=path)
            ok_bag.append(path)
        except:
            print("error")
            error_bag.append(path)
            continue
        data = list(zip(img01, img02, aligned_state_joint, aligned_cmd_joint))
        grouped_data = data
        episode = []
        for i, (img01, img02, state, action) in enumerate(grouped_data):
            episode.append({
                'img01': img01.astype(np.float32),
                'img02': img02.astype(np.float32),
                'state': state,
                'action': action,
            })

        print("episode length", len(episode))
        print("episode keys", episode[0].keys())
        print("episode[0]['img01'].shape", episode[0]['img01'].shape)
        print("episode[0]['img02'].shape", episode[0]['img02'].shape)
        print("episode[0]['state'].shape", episode[0]['state'].shape)
        print("episode[0]['action'].shape", episode[0]['action'].shape)

        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack([x[key] for x in episode])
        replay_buffer.add_episode(data_dict, compressors='disk')
        print(f'saved seed {seed}')

        elapsed_time = time.time() - start_time
        print(f"Time taken for {path}: {elapsed_time:.2f} seconds")
    print("error_bag", error_bag)
    # write error_bag to file
    with open(f"{bag_folder_path}/error_bag.txt", "w") as f:
        for item in error_bag:
            f.write("%s\n" % item)
    
    # write ok_bag to file
    with open(f"{bag_folder_path}/ok_bag.txt", "w") as f:
        for item in ok_bag:
            f.write("%s\n" % item)