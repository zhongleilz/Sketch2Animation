import os
import numpy as np

def convert_kps_joint(motion):
    """
    Convert keypoints by setting specific joints to zero and re-computing intermediate joints.
    """
    MASK_JOINT = [6, 9, 16, 17]
    motion[:, MASK_JOINT, :] = 0.0
    motion[:,16,:] = (motion[:,13,:] + motion[:,18,:]) / 2
    motion[:,17,:] = (motion[:,14,:] + motion[:,19,:]) / 2

    delta = (motion[:,12,:] - motion[:,3,:]) / 3
    motion[:,6,:] = motion[:,3,:] + delta
    motion[:,9,:] = motion[:,3,:] + delta*2
    return motion

def rotate_pose(motion_3d, angle_x=20, angle_y=45):
    """
    Rotate a 3D motion array by given angles around X and Y axes.
    """
    rotation_x = np.radians(angle_x)
    rotation_y = np.radians(angle_y)

    R_y = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_x), -np.sin(rotation_x)],
        [0, np.sin(rotation_x), np.cos(rotation_x)]
    ])

    R = np.dot(R_x, R_y)
    motion_3d_rotated = np.dot(motion_3d, R.T)
    return motion_3d_rotated, R

def project2D(data, angle_x=20, angle_y=30):
    """
    Project 3D data to 2D by applying rotations and selecting XY coordinates.
    """
    data = convert_kps_joint(data)
    motion_2d = np.zeros(data.shape, dtype=np.float32)
    data, Rotation = rotate_pose(data, angle_x=angle_x, angle_y=angle_y)
    motion_2d[:, :, :2] = data[:, :, :2]
    motion_2d[:, :, 2] = 1
    return motion_2d, Rotation
