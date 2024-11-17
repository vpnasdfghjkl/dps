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
# bridge = CvBridge()


CAM_HZ=30
TRAIN_HZ=10
TASK_TIME=1000

def check_folder(CHECK_PIC_SAVE_FOLDER):
    if not os.path.exists(CHECK_PIC_SAVE_FOLDER):
        os.makedirs(CHECK_PIC_SAVE_FOLDER)
    else:
        # 清空文件夹中的所有内容
        shutil.rmtree(CHECK_PIC_SAVE_FOLDER)
        os.makedirs(CHECK_PIC_SAVE_FOLDER)

def adjust_pose_rpy(pose):
    threshold= math.pi
    pre_eef = pose[0][3:6]  
    for i in range(len(pose)):
        for j in range(3): 
            diff = pose[i][3+j] - pre_eef[j]
            if diff > threshold:
                pose[i][3+j] -= 2 * math.pi
            elif diff < -threshold:
                pose[i][3+j] += 2 * math.pi
            pre_eef[j] = pose[i][3+j]
    return pose

def plot_euler_error(cmd_rot_matrix,state_rot_matrix,base_name):
    euler_error = []

    for (c_m,s_m) in zip(cmd_rot_matrix,state_rot_matrix):
        delta_rot_matrix = np.dot(c_m, s_m.T)
        euler_error.append(R.from_matrix(delta_rot_matrix).as_euler('xyz'))
    # plot euler_error
    euler_error = np.array(euler_error[300:])
    fig, axs = plt.subplots(3, 1, figsize=(32, 18))
    fig.suptitle(base_name, fontsize=16)
    for i in range(3):
        axs[i].plot(euler_error[:, i], label=f'error_{i}')
        axs[i].set_title(f"error_{i}")
        axs[i].legend()
    plt.tight_layout()
    save_path = f"{save_plt_folder}/{base_name}_euler_error.png"
    plt.savefig(save_path)
    
def plot_euler_error_direct(cmd_eef_pose,state_eef_pose,base_name):
    # plot euler_error_direct
    euler_error_direct=[]
    for (c_e,s_e) in zip(cmd_eef_pose,state_eef_pose):
        delta_euler = c_e[3:6] - s_e[3:6]
        euler_error_direct.append(delta_euler)
    euler_error_direct = np.array(euler_error_direct[300:])
    fig, axs = plt.subplots(3, 1, figsize=(32, 18))
    fig.suptitle(base_name, fontsize=16)
    for i in range(3):
        axs[i].plot(euler_error_direct[:, i], label=f'error_direct_{i}')
        axs[i].set_title(f"error_direct_{i}")
        axs[i].legend()
    plt.tight_layout()
    save_path = f"{save_plt_folder}/{base_name}_euler_error_direct.png"
    plt.savefig(save_path)
    
def use_rosbag_to_show(bag_folder_path, bag_path):
    save_plt_folder = f"{bag_folder_path}/plt"
    save_lastPic_folder = f"{bag_folder_path}/last_pic"
    check_folder(save_plt_folder)
    check_folder(save_lastPic_folder)
    base_name = os.path.splitext(os.path.basename(bag_path))[0]
    # 读取rosbag文件并提取所需数据
    bag = rosbag.Bag(bag_path, 'r')

    start_time = bag.get_start_time()
    end_time = start_time + TASK_TIME

    cmd_joint=[]
    cmd_joint_time_stamp=[]
    state_joint=[]
    state_joint_time_stamp=[]

    cmd_eef_pose=[]
    cmd_eef_pose_time_stamp=[]
    state_eef_pose=[]
    state_eef_pose_time_stamp=[]

    cmd_hand=[]
    cmd_hand_time_stamp=[]
    state_hand=[]
    state_hand_time_stamp=[]

    img01=[]
    img01_stamp=[]
    img02=[]
    img02_stamp=[]
    
    cmd_rot_matrix = []
    state_rot_matrix = []
    delta_rot_matrix =[]
    
    for topic, msg, t in bag.read_messages(topics=[ 
                                                    '/master/end_right', \
                                                    '/puppet/end_right', \
                                                    '/master/joint_right',\
                                                    '/puppet/joint_right', \
                                                    '/camera_f/color/image_raw',\
                                                    '/camera_r/color/image_raw',
                                                  ]):
        # msg_time = msg.header.stamp.to_sec()  # 将时间戳转换为秒
        # if msg_time > end_time:
        #     break  # 超过时间限制，停止读取
        
        if topic == '/master/joint_right':
            cmd_joint_time_stamp.append(msg.header.stamp)
            cmd_joint.append((msg.position)[:7])

        elif topic == '/puppet/joint_right':
            state_joint_time_stamp.append(msg.header.stamp)
            state_joint.append((msg.position)[:7])
            
        # elif topic=='/master/end_right':
        #     cmd_eef_pose_time_stamp.append(msg.header.stamp)
        #     xyz_dict = msg.pose.position
        #     xyzw_dict = msg.pose.orientation
            
        #     xyz=np.array(xyz_dict[key] for key in xyz_dict)
        #     xyzw=np.array(xyzw_dict[key] for key in xyzw_dict)
            
        #     rotation = R.from_quat(xyzw)
        #     cmd_rot_matrix.append(rotation.as_matrix())  
        #     euler_angles = rotation.as_euler('xyz')
        #     xyzrpy=np.concatenate((xyz,euler_angles))
        #     cmd_eef_pose.append(xyzrpy)

        # elif topic=='/puppet/end_right':
        #     state_eef_pose_time_stamp.append(msg.header.stamp)
        #     xyz=np.array(msg.left_pose.pos_xyz)
        #     xyzw=np.array(msg.left_pose.quat_xyzw)
        #     rotation = R.from_quat(xyzw)
        #     state_rot_matrix.append(rotation.as_matrix())  
        #     # 转换为欧拉角 (默认是 'xyz' 顺序，单位是弧度)
        #     euler_angles = rotation.as_euler('xyz')
        #     xyzrpy=np.concatenate((xyz,euler_angles))
        #     state_eef_pose.append(xyzrpy)

        # elif topic=='/robot_hand_eff':
        #     cmd_hand_time_stamp.append(msg.header.stamp)
        #     left_hand_pose=msg.data
        #     if left_hand_pose[-1]==0:
        #         grip=0
        #     elif left_hand_pose[-1]==90:
        #         grip=1
        #     else:
        #         print("hand pose error")
        #     cmd_hand.append(grip)

        # elif topic=='/robot_hand_position':
        #     state_hand_time_stamp.append(msg.header.stamp)

        #     left_hand_pose=msg.left_hand_position
        #     if left_hand_pose[-1]==0:
        #         grip=0
        #     elif left_hand_pose[-1]==90:
        #         grip=1
        #     else:
        #         print("hand pose error")
        #     state_hand.append(grip)

        elif topic=='/camera_f/color/image_raw':
            img01_stamp.append(msg.header.stamp)
            np_arr = np.frombuffer(msg.data, np.uint8)
            try:
                cv_img = np_arr.reshape((480, 640, 3))  # 这里根据实际图像尺寸调整
                cv_img = cv2.resize(cv_img, (256, 256))
            except ValueError as e:
                print(f"Error reshaping the image: {e}")
                continue
            img01.append(cv_img)
    
            
        elif topic=='/camera_r/color/image_raw':
            img02_stamp.append(msg.header.stamp)
            np_arr = np.frombuffer(msg.data, np.uint8)
            try:
                cv_img = np_arr.reshape((480, 640, 3))  # 这里根据实际图像尺寸调整
                cv_img = cv2.resize(cv_img, (256, 256))
            except ValueError as e:
                print(f"Error reshaping the image: {e}")
                continue
            img02.append(cv_img)
            
    bag.close()
    # cmd_eef_pose=adjust_pose_rpy(cmd_eef_pose)
    # state_eef_pose=adjust_pose_rpy(state_eef_pose)

    # plot_euler_error(cmd_rot_matrix,state_rot_matrix,base_name)
    # plot_euler_error_direct(cmd_eef_pose,state_eef_pose,base_name)
    
    # 安全判断
    if len(cmd_joint) == 0 or len(state_joint) == 0 or len(img01) == 0 or len(img02) == 0:
        print("ROS bag file contains empty data for at least one topic.")
        return

    if len(cmd_joint) < 100 or len(state_joint) < 100 or len(img01) < 100 or len(img02) < 100:
        print("ROS bag file data count is too small (less than 100 data points). Please check again.")
        return
    

    aligned_state_joint = []
    aligned_cmd_joint = []
    aligned_img02 = []
    # aligned_state_hand=[]
    # aligned_cmd_hand=[]
    # aligned_cmd_eef_pose=[]
    # aligned_state_eef_pose=[]
    
    aligned_state_joint_stamp = []
    aligned_cmd_joint_stamp = []
    aligned_img02_stamp = []
    
    drop=2
    img01=img01[drop:-drop]
    # img02=img02[drop:-drop]
    
    img01_stamp=img01_stamp[drop:-drop]
    # img02_stamp=img02_stamp[drop:-drop]
    print("img len before align:")
    print(len(img01),len(img02),len(img01_stamp),len(img02_stamp))
    for stamp in img01_stamp:
        stamp_sec=stamp.to_sec()
        
        idx_s = np.argmin(np.abs(np.array([t.to_sec() for t in state_joint_time_stamp]) - stamp_sec))
        aligned_state_joint.append(state_joint[idx_s])
        aligned_state_joint_stamp.append(state_joint_time_stamp[idx_s])
        
        idx_a = np.argmin(np.abs(np.array([t.to_sec() for t in cmd_joint_time_stamp]) - stamp_sec))
        aligned_cmd_joint.append(cmd_joint[idx_a])
        aligned_cmd_joint_stamp.append(cmd_joint_time_stamp[idx_a])

        idx_img02 = np.argmin(np.abs(np.array([t.to_sec() for t in img02_stamp]) - stamp_sec))
        aligned_img02.append(img02[idx_img02])
        aligned_img02_stamp.append(img02_stamp[idx_img02])
    
    # compare the 4 stamps difference with plot in one picture
    # aligned_state_joint_stamp=np.array([t.to_sec() for t in aligned_state_joint_stamp])
    # aligned_cmd_joint_stamp=np.array([t.to_sec() for t in aligned_cmd_joint_stamp])
    # aligned_img02_stamp=np.array([t.to_sec() for t in aligned_img02_stamp])
    # img01_stamp=np.array([t.to_sec() for t in img01_stamp])
    
    # diff = []
    # prev_t4 = img01_stamp[0]
    # for t1,t2,t3,t4 in zip(aligned_state_joint_stamp,aligned_cmd_joint_stamp,aligned_img02_stamp,img01_stamp):
    #     max_time_diff = max(abs(t1-t2),abs(t1-t3),abs(t1-t4),abs(t2-t3),abs(t2-t4),abs(t3-t4))
    #     mean_time_diff = (abs(t1-t2)+abs(t1-t3)+abs(t1-t4)+abs(t2-t3)+abs(t2-t4)+abs(t3-t4))/6
    #     print(abs(prev_t4-t4),max_time_diff,mean_time_diff,)
    #     prev_t4 = t4
    #     diff.append(max_time_diff)
    # fig, axs = plt.subplots(1, 1, figsize=(32, 18))
    # fig.suptitle(base_name, fontsize=16)
    # axs.plot(diff, label='time_diff')
    # plt.tight_layout()
    # save_path = f"{save_plt_folder}/{base_name}_time_diff.png"
    # plt.savefig(save_path)
    # plt.show()
       

    aligned_cmd_joint = [list(item) for item in aligned_cmd_joint]
    aligned_state_joint = [list(item) for item in aligned_state_joint]
    # aligned_cmd_eef_pose=[list(item) for item in aligned_cmd_eef_pose]
    # aligned_state_eef_pose=[list(item) for item in aligned_state_eef_pose]

    print("all length==============>:\nimg_stamp,aligned_cmd_joint,aligned_state_joint,aligned_cmd_eef_pose,aligned_state_eef_pose,aligned_cmd_hand,aligned_state_hand")
    print(len(img01_stamp),len(aligned_cmd_joint),len(aligned_state_joint),)
    assert len(img01_stamp)==len(aligned_cmd_joint)==len(aligned_state_joint)
    
    # for i in range(len(img01_stamp)):
    #     aligned_cmd_joint[i].append(aligned_cmd_hand[i])
    #     aligned_state_joint[i].append(aligned_state_hand[i])
    #     aligned_cmd_eef_pose[i].append(aligned_cmd_hand[i])
    #     aligned_state_eef_pose[i].append(aligned_state_hand[i])

 # s 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 # a 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

    jump=CAM_HZ//TRAIN_HZ
    img01=img01[::jump]
    img02=aligned_img02[::jump]
    aligned_cmd_joint=np.array(aligned_cmd_joint)[::jump].astype(np.float32)
    aligned_state_joint=np.array(aligned_state_joint)[::jump].astype(np.float32)
    # aligned_cmd_eef_pose=np.array(aligned_cmd_eef_pose)[::jump].astype(np.float32)
    # aligned_state_eef_pose=np.array(aligned_state_eef_pose)[::jump].astype(np.float32)
    # aligned_delta_cmd_eef_pose=None
    

    print("after jump, all length==============>:")
    print(len(img01),len(img02),len(aligned_state_joint),len(aligned_cmd_joint))
    assert len(img01)==len(img02)==len(aligned_cmd_joint)==len(aligned_state_joint)
    
    # 对于夹爪，使用state代替cmd，并把cmd提前一帧
    # aligned_cmd_joint[:,6]=aligned_state_joint[:,6]
    # aligned_cmd_joint[0:-1,6] = aligned_cmd_joint[1:,6]
    

    # aligned_state_eef_pose=aligned_state_eef_pose[1:]
    # aligned_delta_cmd_eef_pose=aligned_cmd_eef_pose[1:]-aligned_cmd_eef_pose[:-1]
    # aligned_delta_cmd_eef_pose[:,6]=aligned_cmd_eef_pose[1:,6]
    # aligned_cmd_eef_pose=aligned_cmd_eef_pose[1:]
    img01=np.array(img01[:])
    img02=np.array(img02[:])
    
    print("after delete firet frame==============>:")
    print(len(img01),len(img02), len(aligned_state_joint),len(aligned_cmd_joint))


    # import matplotlib
    # matplotlib.use('Agg')
    # 创建3行5列的图表并进行比较
    num_plots = min(len(aligned_state_joint[0]), len(aligned_cmd_joint[0]), 15)  # 限制最多只显示15个数据对比
    fig, axs = plt.subplots(3, 5, figsize=(32, 18))
    fig.suptitle(base_name, fontsize=16)
    for i in range(num_plots):
        kuavo_position = [data[i] for data in aligned_cmd_joint]
        robot_q = [data[i] for data in aligned_state_joint]

        # cmd_eef=[data[i] for data in aligned_cmd_eef_pose]
        # state_eef=[data[i] for data in aligned_state_eef_pose]
        # cmd_eef_delta=[data[i] for data in aligned_delta_cmd_eef_pose]
        cmd_joint = [data[i] for data in aligned_cmd_joint]
        state_joint = [data[i] for data in aligned_state_joint]
        
        row = i // 5
        col = i % 5
        axs[row, col].plot(cmd_joint, label='cmd_joint')
        axs[row, col].plot(state_joint, label='state_joint')
        # axs[row, col].plot(cmd_eef, label='/cmd_eef')
        # axs[row, col].plot(state_eef, label='/state_eef')
        # axs[row, col].plot(cmd_eef_delta, label='/cmd_eef_delta')
        axs[row, col].set_title(f"motor {i+1} state")
        axs[row, col].legend()

    exampl_index=50
    print(f"example index {exampl_index}:")
    print(" cmd_joint:",aligned_cmd_joint[exampl_index],
          "\n state_joint:",aligned_state_joint[exampl_index],
        #   "\n aligned_delta_cmd_eef_pose:",aligned_delta_cmd_eef_pose[exampl_index],
        #   "\n state_eef:",aligned_state_eef_pose[exampl_index],
          "\n img01 shape:",img01[exampl_index].shape,
          "\n img02 shape:",img02[exampl_index].shape,
          )   

    plt.tight_layout()
    
    

    # 保存图片
    save_path = f"{save_plt_folder}/{base_name}.png"
    plt.savefig(save_path)

    # 保存最后一张img
    cv2.imwrite(f"{save_lastPic_folder}/{base_name}_img01.png",img01[-1])
    cv2.imwrite(f"{save_lastPic_folder}/{base_name}_img02.png",img02[-1])
    # # 显示图片
    # plt.show()
    assert len(img01)==len(img02)==len(aligned_state_joint)==len(aligned_cmd_joint)
    print("all length==============>:img01,aligned_state_eef_pose,aligned_delta_cmd_eef_pose,aligned_cmd_eef_pose,aligned_state_joint,aligned_cmd_joint")
    print(len(img01),len(img02), len(aligned_state_joint),len(aligned_cmd_joint))   
    return img01, img02, aligned_state_joint, aligned_cmd_joint

if __name__ == "__main__":
    bag_folder_name="pick_place_2024-10-21"
    bag_folder_path="/home/camille/data_utils/rosbag/"+bag_folder_name
    
    save_plt_folder = f"{bag_folder_path}/plt"
    save_lastPic_folder=f"{bag_folder_path}/last_pic"
  
    bagpath=glob.glob(f"{bag_folder_path}/*.bag")
    print(len(bagpath))
    for path in bagpath:
        print("current path",path)
        use_rosbag_to_show(bag_folder_path, path)
