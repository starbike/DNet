import numpy as np
import csv
import re
import cv2
import matplotlib.pyplot as plt 
import math


def to_eularian_angles(q):
    z = q[3]
    y = q[2]
    x = q[1]
    w = q[0]
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)

    return (roll, pitch, yaw)

def read_pfm(file):
    """ Read a pfm file, produce HxW numpy array"""
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = str(bytes.decode(header, encoding='utf-8'))
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # DEY: I don't know why this was there.
    file.close()
    
    return data, scale

# Calculates Rotation Matrix given euler angles.
def euler2R(theta) :
     
    R_x = np.array([[1,                  0,                   0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0,  math.sin(theta[0]), math.cos(theta[0]) ]])
                           
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0,                  1,                  0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
                 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]),  0],
                    [0,                  0,                   1]])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def read_pose(pose_file):
    """convert AirSim poses.txt to transformation matrix between frames
    """
    poses = np.loadtxt(pose_file, skiprows=1, dtype=str)
    global_xyzs = poses[:, 1:4].astype(float)
    global_xyzs = np.expand_dims(global_xyzs, axis=2)
    
    global_quans = poses[:, 4:8].astype(float)
    global_Ts = []
    for i in range(len(poses)):
        global_euler = to_eularian_angles(global_quans[i])
        global_R = euler2R(global_euler)
        
        global_T = np.concatenate((global_R, global_xyzs[i]), 1)

        global_Ts.append(global_T)
    global_Ts = np.array(global_Ts)
    
    gt_global_Ts = np.concatenate(
        (global_Ts, np.zeros((global_Ts.shape[0], 1, 4))), 1)
    gt_global_Ts[:, 3, 3] = 1
    gt_local_poses = []
    for i in range(1, len(gt_global_Ts)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_Ts[i - 1]), gt_global_Ts[i])))

    return np.array(gt_local_poses)

'''
def main():
    pfm_file_path = 'poses.txt'
    gt_local_poses = read_pose(pfm_file_path)
    for i in range(gt_local_poses.shape[0]):
        t = gt_local_poses[i,:,3]
        #print(np.linalg.norm(t))

if __name__ == "__main__":
    main()
'''


