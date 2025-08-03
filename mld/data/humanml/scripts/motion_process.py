import torch

from ..common.quaternion import qinv, qrot
import torch.nn.functional as F


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Converts quaternion to rotation matrix."""
    # Normalize the quaternion
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(-1)
    
    # Compute matrix elements
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz = ty * y, tz * y
    tzz = tz * z

    rot_matrix = torch.stack([
        torch.stack([1 - (tyy + tzz), txy - twz, txz + twy], dim=-1),
        torch.stack([txy + twz, 1 - (txx + tzz), tyz - twx], dim=-1),
        torch.stack([txz - twy, tyz + twx, 1 - (txx + tyy)], dim=-1),
    ], dim=-2)
    return rot_matrix


# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def quaternion_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion to a 6D rotation representation."""
    rot_matrix = quaternion_to_matrix(quat)  # Shape (..., 3, 3)
    # Extract the first two columns of the rotation matrix
    rot_6d = rot_matrix[..., :, :2].reshape(*rot_matrix.shape[:-2], 6)
    return rot_6d


def recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concat root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def extract_rotations(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    # Recover root rotation quaternion and position
    r_rot_quat, _ = recover_root_rot_pos(data)
    
    # Convert root quaternion to 6D rotation
    root_rot_6d = quaternion_to_6d(r_rot_quat)  # Shape (..., 6)
    # Add joint dimension for concatenation
    root_rot_6d = root_rot_6d.unsqueeze(-2)  # Shape (..., 1, 6)
    
    # Extract rot_data indices
    rot_data_start = 4 + (joints_num - 1) * 3
    rot_data_end = rot_data_start + (joints_num - 1) * 6
    rot_data = data[..., rot_data_start:rot_data_end]
    
    # Reshape to (..., joints_num - 1, 6)
    rot_data = rot_data.view(*rot_data.shape[:-1], joints_num - 1, 6)
    
    # Concatenate root rotation with joint rotations
    rot_data = torch.cat([root_rot_6d, rot_data], dim=-2)  # Shape (..., joints_num, 6)
    
    return rot_data