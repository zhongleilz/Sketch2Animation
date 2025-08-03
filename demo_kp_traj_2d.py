import os
import sys
import pickle
import json
import datetime
import logging
import os.path as osp
import numpy as np
import torch
# import cv2
from collections import OrderedDict
from moviepy.editor import VideoFileClip
from omegaconf import OmegaConf

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import set_seed
from mld.data.humanml.utils.plot_script import plot_3d_motion, plot_2d_motion
from mld.utils.temos_utils import remove_padding

# Constants
MASK_JOINT = [6, 9, 16, 17]

####################################
# Helper Functions
####################################

def convert_kps_joint(motion):
    """
    Convert keypoints by setting specific joints to zero and re-computing intermediate joints.
    """
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

def project2D(data, angle_x=20, angle_y=45):
    """
    Project 3D data to 2D by applying rotations and selecting XY coordinates.
    """
    data = convert_kps_joint(data)
    motion_2d = np.zeros(data.shape, dtype=np.float32)
    data, Rotation = rotate_pose(data, angle_x=angle_x, angle_y=angle_y)
    motion_2d[:, :, :2] = data[:, :, :2]
    motion_2d[:, :, 2] = 1
    return motion_2d, Rotation

def split_video_to_jpegs(video_path, output_folder):
    """
    Split a video into individual JPEG frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{count:04d}.jpeg"
        cv2.imwrite(os.path.join(output_folder, filename), frame)
        count += 1
    
    cap.release()
    print(f"Video has been split into {count} images.")

def convert_mp4_to_gif(input_file, output_file, resize=None):
    """
    Convert an MP4 video to a GIF.
    """
    clip = VideoFileClip(input_file)
    # if resize:
    #     clip = clip.resize(resize)
    clip.write_gif(output_file, fps=20)

def load_json(file_path):
    """
    Load a JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def load_example_input(text_path: str) -> tuple:
    """
    Load example input from a text file with format:
    <length> <text>
    """
    with open(text_path, "r") as f:
        lines = f.readlines()

    texts, lens = [], []
    for line in lines:
        s = line.strip()
        s_l = s.split(" ")[0]
        s_t = s[(len(s_l) + 1):]
        lens.append(int(s_l))
        texts.append(s_t)
    return texts, lens

def setup_logging_and_directories(cfg):
    """
    Setup logging and output directories.
    """
    name_time_str = osp.join(cfg.NAME, "demo_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    output_dir = osp.join(cfg.TEST_FOLDER, name_time_str)
    vis_dir = osp.join(output_dir, 'samples')
    os.makedirs(output_dir, exist_ok=False)
    os.makedirs(vis_dir, exist_ok=False)

    steam_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(output_dir, 'output.log'))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[steam_handler, file_handler])
    logger = logging.getLogger(__name__)
    OmegaConf.save(cfg, osp.join(output_dir, 'config.yaml'))

    return logger, output_dir, vis_dir

def identify_model_type(state_dict, cfg, logger):
    """
    Identify whether the model is VAE, MLD, LCM, and/or ControlNet.
    Update cfg accordingly.
    """
    # Check if VAE-based
    is_vae = 'vae.skel_embedding.weight' in state_dict
    logger.info(f'Is VAE: {is_vae}')

    # Check if MLD-based
    is_mld = 'denoiser.time_embedding.linear_1.weight' in state_dict
    logger.info(f'Is MLD: {is_mld}')

    # Check if LCM-based
    lcm_key = 'denoiser.time_embedding.cond_proj.weight'  
    is_lcm = lcm_key in state_dict
    if is_lcm:
        time_cond_proj_dim = state_dict[lcm_key].shape[1]
        cfg.model.controlnet.params.time_cond_proj_dim = time_cond_proj_dim
        cfg.model.denoiser.params.time_cond_proj_dim = time_cond_proj_dim
    logger.info(f'Is LCM: {is_lcm}')

    # Check if ControlNet-based
    cn_key = "controlnet.controlnet_cond_embedding.0.weight"
    is_controlnet = cn_key in state_dict
    cfg.model.is_controlnet = is_controlnet
    logger.info(f'Is Controlnet: {is_controlnet}')

def extract_substate_dict(pre_state_dict, prefix):
    """
    Extract a substate dictionary for VAE or denoiser from a pre_state_dict.
    """
    sub_dict = OrderedDict()
    for k, v in pre_state_dict.items():
        if k.startswith(prefix + "."):
            name = k.replace(prefix + ".", "")
            sub_dict[name] = v
    return sub_dict

####################################
# Main Function
####################################

def main():
    cfg = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.TRAIN.SEED_VALUE)

    logger, output_dir, vis_dir = setup_logging_and_directories(cfg)

    # Load state dicts
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    pre_state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]

    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    identify_model_type(state_dict, cfg, logger)

    # Extract partial dicts if needed
    denoiser_dict = extract_substate_dict(pre_state_dict, "denoiser")
    vae_dict = extract_substate_dict(pre_state_dict, "vae")

    dataset = get_dataset(cfg, phase="test")
    model = MLD(cfg, dataset).to(device)
    model.eval()
    model.load_state_dict(state_dict, strict=True)

    # Load normalization stats
    mean_pose = torch.tensor(np.load('/home/lei/dataset_all/Mean_pos.npy')).cuda()
    std_pose = torch.tensor(np.load('/home/lei/dataset_all/Std_pos.npy')).cuda()
    raw_mean = np.load('/home/lei/MotionLCM2/datasets/humanml_spatial_norm/Mean_raw.npy')
    raw_std = np.load('/home/lei/MotionLCM2/datasets/humanml_spatial_norm/Std_raw.npy')

    joint_ids = np.array([15])  # pelvis

    length = 196
    control_full = np.zeros((1, length, 22, 3), dtype=np.float32)
    control_full_2d = np.zeros((1, length, 22, 3), dtype=np.float32)

    # Load motion
    motion_path = "/home/lei/dataset_all/new_joints/011988.npy"
    # motion_path = "./test_data_old/Chicken_FW_00_pos.npy"
    joints_3d = np.load(motion_path)#[:-10]
    motion_len = len(joints_3d)
    # Center motion
    joints_3d[..., 0] -= joints_3d[0:1, 0:1, 0]
    joints_3d[..., 2] -= joints_3d[0:1, 0:1, 2]

    # Project to 2D
    joints_2d, Rotation = project2D(joints_3d)

    # Prepare torch tensors
    def normalize(j):
        return (j - torch.from_numpy(raw_mean).cuda()) / torch.from_numpy(raw_std).cuda()

    joints_3d_t = torch.from_numpy(joints_3d.reshape(joints_3d.shape[0], -1)).cuda().double()
    joints_2d_t = torch.from_numpy(joints_2d.reshape(joints_2d.shape[0], -1)).cuda().double()

    joints_3d_t = normalize(joints_3d_t.unsqueeze(0)).reshape(-1,22,3)
    joints_2d_t = normalize(joints_2d_t.unsqueeze(0)).reshape(-1,22,3)

    # Fill in control signals
    # control_full[0, 0:motion_len, joint_id[0], :] = joints_3d_t.cpu().numpy()[0:motion_len, joint_id[0], :]
    # control_full_2d[0, 0:motion_len, joint_id[0], :] = joints_2d_t.cpu().numpy()[0:motion_len, joint_id[0], :]
    for joint_id in joint_ids:
        control_full[0, 0:motion_len, joint_id, :] = joints_3d_t.cpu().numpy()[0:motion_len, joint_id, :]
    
    for joint_id in joint_ids:
        control_full_2d[0, 0:motion_len, joint_id, :] = joints_2d_t.cpu().numpy()[0:motion_len, joint_id, :]

    control_full = control_full.reshape((1, length, -1))
    control_full_2d = control_full_2d.reshape((1, length, -1))

    # Load text
    test_data = load_json("./demo/demo.json")
    text = [test_data["caption"]]

    # Re-load and re-project (for visualization consistency)
    joints_3d = np.load(motion_path)#[:-10]
    joints_3d[..., 0] -= joints_3d[0:1, 0:1, 0]
    joints_3d[..., 2] -= joints_3d[0:1, 0:1, 2]

    joints_2d, Rotation = project2D(joints_3d)
    joints_3d = torch.from_numpy(joints_3d).cuda().double().reshape(joints_3d.shape[0], -1).unsqueeze(0)
    joints_2d = torch.from_numpy(joints_2d).cuda().double().reshape(joints_2d.shape[0], -1).unsqueeze(0)

    joints_3d = normalize(joints_3d)
    joints_2d = normalize(joints_2d)

    motion_idx = [-1]
    T = joints_3d.shape[1]
    mask_array = np.zeros(T, dtype=int)
    mask_array[motion_idx] = 1
    mask_array = mask_array[:, np.newaxis]

    pose = joints_3d.float().cuda()
    pose_2d = joints_2d.float().cuda()
    hint = torch.from_numpy(control_full).cuda()
    hint_2d = torch.from_numpy(control_full_2d).cuda()
    mask = torch.from_numpy(mask_array).cuda().unsqueeze(0)
    rotation = torch.from_numpy(Rotation).cuda().float()

    batch = {
        "length": [length],
        "text": text,
        "motion_idx": [motion_idx],
        "pose": pose,
        "pose_2d": pose_2d,
        "hint": hint,
        "pose_mask": mask,
        "hint_2d": hint_2d,
        "rotation": rotation
    }

    cfg.replication = 5

    # Denormalize for ground truth visualization
    joints_3d_dn = joints_3d * torch.from_numpy(raw_std).cuda() + torch.from_numpy(raw_mean).cuda()
    joints_3d_dn = joints_3d_dn[:, motion_idx[0], :].reshape(22,3)[np.newaxis, :, :]
    joints_2d_dn = joints_2d * torch.from_numpy(raw_std).cuda() + torch.from_numpy(raw_mean).cuda()
    joints_2d_dn = joints_2d_dn[:, motion_idx[0], :].reshape(22,3)[np.newaxis, :, :]

    # Inference and visualization
    for rep_i in range(cfg.replication):
        with torch.no_grad():
            joints_pred, _ = model.forward_gmld_sequence_2d_keypose_traj(batch)

        num_samples = len(joints_pred)
        batch_id = 0

        # Process hints if available
        if 'hint' in batch:
            hint = batch['hint']
            mask_hint = hint.view(hint.shape[0], hint.shape[1], model.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint = model.datamodule.denorm_spatial(hint)
            hint = hint.view(hint.shape[0], hint.shape[1], model.njoints, 3) * mask_hint
            # hint = remove_padding(hint, lengths=length)

            hint_2d = batch['hint_2d']
            mask_hint_2d = hint_2d.view(hint_2d.shape[0], hint_2d.shape[1], model.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint_2d = model.datamodule.denorm_spatial(hint_2d)
            hint_2d = hint_2d.view(hint_2d.shape[0], hint_2d.shape[1], model.njoints, 3) * mask_hint_2d
            # hint_2d = remove_padding(hint_2d, lengths=length)
        else:
            hint = None
            hint_2d = None

        for i in range(num_samples):


            res = {
                'joints': joints_pred[i][:motion_len].detach().cpu().numpy(),
                'text': text[i],
                'length': length,
                'hint': hint[i][:motion_len].detach().cpu().numpy() if hint is not None else None,
                'hint_2d': hint_2d[i][:motion_len].detach().cpu().numpy() if hint_2d is not None else None,
                'joints_3d_dn': joints_3d_dn[:motion_len].cpu().numpy() if joints_3d_dn is not None else None,
                'joints_2d_dn': joints_2d_dn[:motion_len].cpu().numpy() if joints_2d_dn is not None else None
            }

            pkl_path = osp.join(vis_dir, f"batch_id_{batch_id}_sample_id_{i}_length_{length}_rep_{rep_i}.pkl")
            pkl_2d_path = osp.join(vis_dir, f"batch_id_{batch_id}_sample_id_{i}_length_{length}_rep_{rep_i}_2d.pkl")
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(res, f)
            logger.info(f"Motions are generated here:\n{pkl_path}")

            # Generate 2D projection for visualization
            gen_joints_2d, _ = project2D(joints_pred[i].detach().cpu().numpy())

            if not cfg.no_plot:
                plot_3d_motion(
                    pkl_path.replace('.pkl', '.mp4'),
                    joints_pred[i][:motion_len].detach().cpu().numpy(),
                    text[i],
                    fps=20,
                    joint_id=joint_ids[0],
                    hint=res['hint'] if hint is not None else None,
                    gt_frames=joints_3d_dn.cpu()
                )

                plot_2d_motion(
                    pkl_2d_path.replace('.pkl', '.mp4'),
                    joints=gen_joints_2d[:motion_len],
                    hint=hint_2d[i].detach().cpu().numpy() if hint_2d is not None else None,
                    gt_frames_2d=joints_2d_dn.cpu(),
                    title=text[i],
                    fps=20,
                    joint_id=joint_ids[0]
                )

                convert_mp4_to_gif(pkl_path.replace('.pkl', '.mp4'), pkl_path.replace('.pkl', '.gif'))
                convert_mp4_to_gif(pkl_2d_path.replace('.pkl', '.mp4'), pkl_2d_path.replace('.pkl', '.gif'))


if __name__ == "__main__":
    main()
