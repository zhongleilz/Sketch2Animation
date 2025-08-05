import random
import logging
import codecs as cs
from os.path import join as pjoin
import os

import numpy as np
from rich.progress import track

import torch
from torch.utils import data

from mld.data.humanml.scripts.motion_process import recover_from_ric
from .utils.word_vectorizer import WordVectorizer
from .utils.load_json import load_json, save_json, process

logger = logging.getLogger(__name__)
mask_joint = [6,9,16,17]

def adjust_smpl_proportions(positions, spine_scale=1.0, neck_head_scale=1.0, arm_scale=1.0, thigh_scale=1.0, calf_scale=1.0, shoulder_width_scale=1.0, shoulder_height_scale=1.0):
    """
    Adjust SMPL body proportions with separate scaling factors for spine, neck+head, arms, thighs, calves, and shoulders.

    :param positions: numpy array of shape (len, 22, 3), keypoint positions.
    :param spine_scale: scaling factor for spine length (spine1 -> spine3).
    :param neck_head_scale: scaling factor for neck + head length (neck -> head).
    :param arm_scale: scaling factor for arm length.
    :param thigh_scale: scaling factor for thigh length (hip -> knee).
    :param calf_scale: scaling factor for calf length (knee -> ankle).
    :param shoulder_width_scale: scaling factor for shoulder width (spine3 -> shoulders horizontally).
    :param shoulder_height_scale: scaling factor for shoulder height (spine3 -> shoulders vertically).
    :return: numpy array of adjusted keypoint positions.
    """
    # Make a copy to avoid modifying the original data
    adjusted_positions = positions.copy()

    if thigh_scale != 1.0:
        calf_scale = thigh_scale#(thigh_scale / 1.2)
    

    # SMPL joint indices based on provided order
    pelvis_idx = 0
    spine1_idx = 3
    spine2_idx = 6
    spine3_idx = 9
    neck_idx = 12
    head_idx = 15
    left_shoulder_idx = 16
    right_shoulder_idx = 17
    left_elbow_idx = 18
    right_elbow_idx = 19
    left_wrist_idx = 20
    right_wrist_idx = 21
    left_hip_idx = 1
    right_hip_idx = 2
    left_knee_idx = 4
    right_knee_idx = 5
    left_ankle_idx = 7
    right_ankle_idx = 8
    left_foot_idx = 10
    right_foot_idx = 11

    for t in range(positions.shape[0]):
        # Adjust spine (pelvis -> spine3)
        pelvis_to_spine1 = adjusted_positions[t, spine1_idx] - adjusted_positions[t, pelvis_idx]
        spine1_to_spine2 = adjusted_positions[t, spine2_idx] - adjusted_positions[t, spine1_idx]
        spine2_to_spine3 = adjusted_positions[t, spine3_idx] - adjusted_positions[t, spine2_idx]

        adjusted_positions[t, spine1_idx] = adjusted_positions[t, pelvis_idx] + pelvis_to_spine1 * spine_scale
        adjusted_positions[t, spine2_idx] = adjusted_positions[t, spine1_idx] + spine1_to_spine2 * spine_scale
        adjusted_positions[t, spine3_idx] = adjusted_positions[t, spine2_idx] + spine2_to_spine3 * spine_scale

        # Adjust shoulders (spine3 -> shoulders)
        spine3_to_left_shoulder = adjusted_positions[t, left_shoulder_idx] - adjusted_positions[t, spine3_idx]
        spine3_to_right_shoulder = adjusted_positions[t, right_shoulder_idx] - adjusted_positions[t, spine3_idx]

        adjusted_positions[t, left_shoulder_idx] = adjusted_positions[t, spine3_idx] + \
            spine3_to_left_shoulder * [shoulder_width_scale, shoulder_height_scale, 1.0]
        adjusted_positions[t, right_shoulder_idx] = adjusted_positions[t, spine3_idx] + \
            spine3_to_right_shoulder * [shoulder_width_scale, shoulder_height_scale, 1.0]

        # Adjust neck + head (spine3 -> neck -> head)
        spine3_to_neck = adjusted_positions[t, neck_idx] - adjusted_positions[t, spine3_idx]
        neck_to_head = adjusted_positions[t, head_idx] - adjusted_positions[t, neck_idx]

        adjusted_positions[t, neck_idx] = adjusted_positions[t, spine3_idx] + spine3_to_neck * neck_head_scale
        adjusted_positions[t, head_idx] = adjusted_positions[t, neck_idx] + neck_to_head * neck_head_scale

        # Adjust arms, thighs, and calves (same logic as before)
        # [Same as previously shown logic for arms, thighs, and calves]
        # Adjust arm lengths
        # Left arm
        shoulder_to_elbow = adjusted_positions[t, left_elbow_idx] - adjusted_positions[t, left_shoulder_idx]
        elbow_to_wrist = adjusted_positions[t, left_wrist_idx] - adjusted_positions[t, left_elbow_idx]
        adjusted_positions[t, left_elbow_idx] = adjusted_positions[t, left_shoulder_idx] + shoulder_to_elbow * arm_scale
        adjusted_positions[t, left_wrist_idx] = adjusted_positions[t, left_elbow_idx] + elbow_to_wrist * arm_scale

        # Right arm
        shoulder_to_elbow = adjusted_positions[t, right_elbow_idx] - adjusted_positions[t, right_shoulder_idx]
        elbow_to_wrist = adjusted_positions[t, right_wrist_idx] - adjusted_positions[t, right_elbow_idx]
        adjusted_positions[t, right_elbow_idx] = adjusted_positions[t, right_shoulder_idx] + shoulder_to_elbow * arm_scale
        adjusted_positions[t, right_wrist_idx] = adjusted_positions[t, right_elbow_idx] + elbow_to_wrist * arm_scale

        # Adjust left leg
        # Thigh adjustment (hip -> knee)
        hip_to_knee = adjusted_positions[t, left_knee_idx] - adjusted_positions[t, left_hip_idx]
        adjusted_positions[t, left_knee_idx] = adjusted_positions[t, left_hip_idx] + hip_to_knee * thigh_scale

        # Calf adjustment (knee -> ankle)
        knee_to_ankle = adjusted_positions[t, left_ankle_idx] - adjusted_positions[t, left_knee_idx]
        adjusted_positions[t, left_ankle_idx] = adjusted_positions[t, left_knee_idx] + knee_to_ankle * calf_scale

        # Foot correction (left)
        ankle_to_foot = adjusted_positions[t, left_foot_idx] - adjusted_positions[t, left_ankle_idx]
        adjusted_positions[t, left_foot_idx] = adjusted_positions[t, left_ankle_idx] + ankle_to_foot

        # Adjust right leg
        # Thigh adjustment (hip -> knee)
        hip_to_knee = adjusted_positions[t, right_knee_idx] - adjusted_positions[t, right_hip_idx]
        adjusted_positions[t, right_knee_idx] = adjusted_positions[t, right_hip_idx] + hip_to_knee * thigh_scale

        # Calf adjustment (knee -> ankle)
        knee_to_ankle = adjusted_positions[t, right_ankle_idx] - adjusted_positions[t, right_knee_idx]
        adjusted_positions[t, right_ankle_idx] = adjusted_positions[t, right_knee_idx] + knee_to_ankle * calf_scale

        # Foot correction (right)
        ankle_to_foot = adjusted_positions[t, right_foot_idx] - adjusted_positions[t, right_ankle_idx]
        adjusted_positions[t, right_foot_idx] = adjusted_positions[t, right_ankle_idx] + ankle_to_foot

    return adjusted_positions


def rotate_pose(motion_3d, angle_x=20, angle_y=45):
    rotation_x=np.radians(angle_x)
    rotation_y=np.radians(angle_y)

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
        
    return motion_3d_rotated,R


class Text2MotionDatasetV2_Proj2d(data.Dataset):

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        w_vectorizer: WordVectorizer,
        max_motion_length: int,
        min_motion_length: int,
        max_text_len: int,
        unit_length: int,
        motion_dir: str,
        text_dir: str,
        fps: int,
        tiny: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        maxdata = 10 if tiny else 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if len(motion) < self.min_motion_length or len(motion) >= self.max_motion_length:
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < self.min_motion_length or \
                                        len(n_motion) >= self.max_motion_length:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if not tiny:
            logger.info(f"Reading {len(self.id_list)} motions from {split_file}.")
            logger.info(f"Total {len(name_list)} motions are used.")
            logger.info(f"{bad_count} motion sequences not within the length range of "
                        f"[{self.min_motion_length}, {self.max_motion_length}) are filtered out.")

        self.mean = mean
        self.std = std

        self.mode = None
        model_params = kwargs['model_kwargs']
        if 'is_controlnet' in model_params and model_params.is_controlnet is True:
            if 'test' in split_file or 'val' in split_file:
                self.mode = 'eval'
            else:
                self.mode = 'train'

            self.t_ctrl = model_params.is_controlnet_temporal
            spatial_norm_path = './datasets/humanml_spatial_norm'
            self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
            self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

            self.training_control_joint = np.array(model_params.training_control_joint)
            self.testing_control_joint = np.array(model_params.testing_control_joint)

            self.training_density = model_params.training_density
            self.testing_density = model_params.testing_density

        self.data_dict = data_dict
        self.nfeats = data_dict[name_list[0]]["motion"].shape[1]
        self.name_list = name_list

        self.mean_2d = np.load('/home/lei/dataset_all/Mean_pos_2d.npy')
        self.std_2d = np.load('/home/lei/dataset_all/Std_pos_2d.npy')


    def __len__(self) -> int:
        return len(self.name_list)

    def random_mask(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        choose_joint = self.testing_control_joint

        length = joints.shape[0]
        density = self.testing_density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        # joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        # joints = joints * mask_seq
        return joints,mask_seq
    
    def random_mask_multi_joint(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        # choose_joint = self.testing_control_joint

        num_joints = len(self.testing_control_joint)
        num_joints_control = np.random.randint(1, num_joints + 1)  # 随机选择 1-3 个关节
        choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
        choose_joint = self.testing_control_joint[choose_joint]

        length = joints.shape[0]
        density = self.testing_density
        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            if density in [1, 2, 5]:
                choose_seq_num = density
            else:
                choose_seq_num = int(length * density / 100)

            if self.t_ctrl:
                choose_seq = np.arange(0, choose_seq_num)
            else:
                choose_seq = np.random.choice(length, choose_seq_num, replace=False)
                choose_seq.sort()

            mask_seq[choose_seq, cj] = True        

        # normalize
        # joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        # joints = joints * mask_seq
        return joints,mask_seq

    def random_mask_train_multi_joint(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        if self.t_ctrl:
            choose_joint = self.training_control_joint
        else:
            num_joints = len(self.training_control_joint)
            num_joints_control = np.random.randint(1, num_joints + 1)  
            choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
            choose_joint = self.training_control_joint[choose_joint]

        length = joints.shape[0]

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            if self.training_density == 'random':
                choose_seq_num = np.random.choice(length - 1, 1) + 1
            else:
                choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

            if self.t_ctrl:
                choose_seq = np.arange(0, choose_seq_num)
            else:
                choose_seq = np.random.choice(length, choose_seq_num, replace=False)
                choose_seq.sort()

            mask_seq[choose_seq, cj] = True

        # joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        # joints = joints * mask_seq
        return joints,mask_seq


    def random_mask_train(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        if self.t_ctrl:
            choose_joint = self.training_control_joint
        else:
            num_joints = len(self.training_control_joint)
            num_joints_control = 1
            choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
            choose_joint = self.training_control_joint[choose_joint]

        length = joints.shape[0]

        if self.training_density == 'random':
            choose_seq_num = np.random.choice(length - 1, 1) + 1
        else:
            choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        # joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        # joints = joints * mask_seq
        return joints, mask_seq

    def convertKpsJoint(self,motion):

        motion[:,mask_joint,:] = 0.0

        motion[:,16,:] = (motion[:,13,:] + motion[:,18,:]) / 2
        motion[:,17,:] = (motion[:,14,:] + motion[:,19,:]) / 2

        delta = (motion[:,12,:] - motion[:,3,:]) / 3
        motion[:,6,:] = motion[:,3,:] + delta
        motion[:,9,:] = motion[:,3,:] + delta*2

        return motion

    def scale_and_translate(self, motion_2d, scale_range=(0.8, 1.2), translate_range=(-2, 2)):

        scale = np.random.uniform(*scale_range)
        translate_x = np.random.uniform(*translate_range)
        translate_y = np.random.uniform(*translate_range)

        transformed_motion = motion_2d * scale
        transformed_motion[..., 0] += translate_x
        transformed_motion[..., 1] += translate_y
        return transformed_motion

    def add_joint_noise(self,motion_2d, noise_level=0.01):
        noise = np.random.normal(scale=noise_level, size=motion_2d.shape)
        noisy_motion = motion_2d + noise
        return noisy_motion


    def jitter_trajectory(self,motion_2d, jitter_level=0.01):

        jitter = np.random.uniform(-jitter_level, jitter_level, size=motion_2d.shape)
        jittered_motion = motion_2d + jitter
        return jittered_motion

    def project2D(self, data):
        # (128, 22, 3)
        data = self.convertKpsJoint(data)

        motion_2d = np.zeros(data.shape, dtype=np.float32)

        # angle_x = np.random.uniform(15, 30)  # ±15°
        # angle_y = np.random.uniform(30, 45)  # ±30°

        if self.mode == 'train':
            angle_x = np.random.uniform(0, 30)  # ±15°
            angle_y = np.random.uniform(-45, 45)
        else:
            angle_x = np.random.uniform(15, 30)  # ±15°
            angle_y = np.random.uniform(30, 45)

       
        data, Rotation = rotate_pose(data,angle_x=angle_x,angle_y=angle_y )


        motion_2d[:,:,:2] = data[:,:,:2]
        motion_2d[:,:,2] = 1

        return motion_2d, Rotation
    
    def augment2D(self, motion_2d, noise_prob=0.6, jitter_prob=0.6):

        motion_2d = self.scale_and_translate(motion_2d)

        if random.random() < noise_prob:
            motion_2d = self.add_joint_noise(motion_2d)

        if random.random() < jitter_prob:
            motion_2d = self.jitter_trajectory(motion_2d)

        motion_2d[:,:,2] = 1
        return motion_2d

    def __getitem__(self, idx: int) -> tuple:
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        hint = None
        if self.mode is not None:
            n_joints = 22 if motion.shape[-1] == 263 else 21
            # hint is global position of the controllable joints
            joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
            joints = joints.numpy()

            # control any joints at any time
            if self.mode == 'train':
                hint, mask_seq = self.random_mask_train_multi_joint(joints, n_joints)
                # hint,mask_seq = self.random_mask_train(joints, n_joints)
            else:
                hint, mask_seq = self.random_mask_multi_joint(joints, n_joints)
                # hint,mask_seq = self.random_mask(joints, n_joints)

            # hint = hint.reshape(hint.shape[0], -1)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        hint_2d, Rotation = self.project2D(joints)

        scale_ranges = {
            "spine_scale": (0.6, 1.6),       
            "neck_head_scale": (0.6, 1.6), 
            "arm_scale": (0.6, 1.6),       
            "thigh_scale": (0.6, 1.6),      
            "shoulder_width_scale": (0.7, 1.2),  
        }

        random_params = {key: np.random.uniform(low, high) for key, (low, high) in scale_ranges.items()}


        joints_aug = adjust_smpl_proportions(joints,spine_scale=random_params['spine_scale'],neck_head_scale=random_params['neck_head_scale'],
                                            arm_scale=random_params['arm_scale'], thigh_scale=random_params['thigh_scale'],shoulder_width_scale=random_params['shoulder_width_scale'])

        joints_2d_aug = np.dot(joints_aug, Rotation.T)

        hint_2d_a = hint_2d#self.augment2D(hint_2d)


        hint_2d_b = self.augment2D(hint_2d) #self.augment2D(joints_2d_aug)#self.augment2D(hint_2d)

        first_true_index = np.argmax(np.any(mask_seq, axis=(1, 2)))

        # mask_pose = np.zeros_like(mask_seq, dtype=int) 
        mask_pose = np.zeros(len(mask_seq),dtype=bool)

        mask_pose[first_true_index] = 1

        mask_pose = mask_pose[:, np.newaxis]

        mask_seq[first_true_index] = 1
        

        hint = (hint - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        # pose_3d = hint * mask_pose

        pose_3d = hint.copy()
        pose_3d = pose_3d.reshape(pose_3d.shape[0], -1)
        pose_3d = pose_3d * mask_pose

        hint = hint * mask_seq
        hint = hint.reshape(hint.shape[0], -1)

        hint_2d_a[...,:2] = (hint_2d_a[...,:2] - self.raw_mean.reshape(n_joints, 3)[...,:2]) / self.raw_std.reshape(n_joints, 3)[...,:2]
        hint_2d_b[...,:2] = (hint_2d_b[...,:2] - self.raw_mean.reshape(n_joints, 3)[...,:2]) / self.raw_std.reshape(n_joints, 3)[...,:2]
        
        pose_2d = hint_2d_a.copy()
        pose_2d = pose_2d.reshape(pose_2d.shape[0], -1)
        pose_2d = pose_2d * mask_pose

        pose_2d_b = hint_2d_b.copy()
        pose_2d_b = pose_2d_b.reshape(pose_2d_b.shape[0], -1)
        pose_2d_b = pose_2d_b * mask_pose

        hint_2d_a = hint_2d_a * mask_seq
        hint_2d_a = hint_2d_a.reshape(hint_2d_a.shape[0], -1)

        hint_2d_b = hint_2d_b * mask_seq
        hint_2d_b = hint_2d_b.reshape(hint_2d_b.shape[0], -1)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            hint_2d_a,
            hint_2d_b,
            pose_2d,
            pose_3d,
            mask_pose,
            Rotation,
            pose_2d_b,
            hint
        )



class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        w_vectorizer: WordVectorizer,
        max_motion_length: int,
        min_motion_length: int,
        max_text_len: int,
        unit_length: int,
        motion_dir: str,
        text_dir: str,
        fps: int,
        tiny: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        maxdata = 10 if tiny else 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if len(motion) < self.min_motion_length or len(motion) >= self.max_motion_length:
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < self.min_motion_length or \
                                        len(n_motion) >= self.max_motion_length:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if not tiny:
            logger.info(f"Reading {len(self.id_list)} motions from {split_file}.")
            logger.info(f"Total {len(name_list)} motions are used.")
            logger.info(f"{bad_count} motion sequences not within the length range of "
                        f"[{self.min_motion_length}, {self.max_motion_length}) are filtered out.")

        self.mean = mean
        self.std = std

        self.mode = None
        model_params = kwargs['model_kwargs']
        if 'is_controlnet' in model_params and model_params.is_controlnet is True:
            if 'test' in split_file or 'val' in split_file:
                self.mode = 'eval'
            else:
                self.mode = 'train'

            self.t_ctrl = model_params.is_controlnet_temporal
            spatial_norm_path = './datasets/humanml_spatial_norm'
            self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
            self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

            self.training_control_joint = np.array(model_params.training_control_joint)
            self.testing_control_joint = np.array(model_params.testing_control_joint)

            self.training_density = model_params.training_density
            self.testing_density = model_params.testing_density

        self.data_dict = data_dict
        self.nfeats = data_dict[name_list[0]]["motion"].shape[1]
        self.name_list = name_list

    def __len__(self) -> int:
        return len(self.name_list)

    def random_mask(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        choose_joint = self.testing_control_joint

        length = joints.shape[0]
        density = self.testing_density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask_multi_joint(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        choose_joint = self.testing_control_joint

        length = joints.shape[0]
        density = self.testing_density
        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            if density in [1, 2, 5]:
                choose_seq_num = density
            else:
                choose_seq_num = int(length * density / 100)

            if self.t_ctrl:
                choose_seq = np.arange(0, choose_seq_num)
            else:
                choose_seq = np.random.choice(length, choose_seq_num, replace=False)
                choose_seq.sort()

            mask_seq[choose_seq, cj] = True        

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train_multi_joint(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        # 随机选择 1-3 个关节
        if self.t_ctrl:
            choose_joint = self.training_control_joint
        else:
            num_joints = len(self.training_control_joint)
            num_joints_control = np.random.randint(1, num_joints + 1)  # 随机选择 1-3 个关节
            choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
            choose_joint = self.training_control_joint[choose_joint]

        length = joints.shape[0]

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            if self.training_density == 'random':
                choose_seq_num = np.random.choice(length - 1, 1) + 1
            else:
                choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

            if self.t_ctrl:
                choose_seq = np.arange(0, choose_seq_num)
            else:
                choose_seq = np.random.choice(length, choose_seq_num, replace=False)
                choose_seq.sort()

            mask_seq[choose_seq, cj] = True

        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints


    def random_mask_train(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        if self.t_ctrl:
            choose_joint = self.training_control_joint
        else:
            num_joints = len(self.training_control_joint)
            num_joints_control = 1
            choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
            choose_joint = self.training_control_joint[choose_joint]

        length = joints.shape[0]

        if self.training_density == 'random':
            choose_seq_num = np.random.choice(length - 1, 1) + 1
        else:
            choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def __getitem__(self, idx: int) -> tuple:
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        hint = None
        if self.mode is not None:
            n_joints = 22 if motion.shape[-1] == 263 else 21
            # hint is global position of the controllable joints
            joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
            joints = joints.numpy()

            # control any joints at any time
            if self.mode == 'train':
                hint = self.random_mask_train_multi_joint(joints, n_joints)
                # hint = self.random_mask_train(joints, n_joints)
            else:
                hint = self.random_mask_multi_joint(joints, n_joints)
                # hint = self.random_mask(joints, n_joints)

            hint = hint.reshape(hint.shape[0], -1)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            hint
        )


class Text2MotionDatasetKeyPoseTrejControl(data.Dataset):

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        w_vectorizer: WordVectorizer,
        max_motion_length: int,
        min_motion_length: int,
        max_text_len: int,
        unit_length: int,
        motion_dir: str,
        text_dir: str,
        # fps: int,
        tiny: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []

        split_dir = os.path.dirname(split_file)
        split_base = os.path.basename(split_file).split(".")[0]
        split_file = os.path.join(split_dir,split_base + "_humanml.txt")#_humanml _humanml_100style _100STYLE

        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if "test" in split_file:
            graph = load_json("/home/lei/Sketch2Anim/datasets/humanml3d/new_test_data_with_motionidx.json")
        if "train" in split_file:
            graph = load_json("/home/lei/Sketch2Anim/datasets/humanml3d/new_train_data_with_motionidx.json")
        
        id_list = graph.keys()   
        self.id_list = id_list

        maxdata = 10 if tiny else 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            # try:
            motion = np.load(pjoin(motion_dir, name + ".npy"))

            pose_dir = motion_dir.replace("new_joint_vecs","new_joints")
            pose = np.load(pjoin(pose_dir, name + ".npy"))
            if pose.ndim != 3:
                continue

            pose[..., 0] -= pose[:, 0:1, 0]
            pose[..., 2] -= pose[:, 0:1, 2]

            pose = pose.reshape(pose.shape[0], -1)

            if len(motion) < self.min_motion_length or len(motion) >= self.max_motion_length:
                bad_count += 1
                continue
            text_data = []
            flag = True
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)

            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                
                    text_dict["caption"] = caption
                    text_dict["tokens"] = tokens
                    text_data.append(text_dict)
                if flag:
                    
                    data_dict[name] = {
                        "motion": motion,
                        "pose":pose,
                        "length": len(motion),
                        "text": text_data,

                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1


        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))


        self.mean = mean
        self.std = std

        self.mode = None
        model_params = kwargs['model_kwargs']
        # if 'is_controlnet' in model_params and model_params.is_controlnet is True:
        if 'test' in split_file or 'val' in split_file:
            self.mode = 'eval'
        else:
            self.mode = 'train'

        self.t_ctrl = model_params.is_controlnet_temporal
        spatial_norm_path = './datasets/humanml_spatial_norm'
        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

        self.pos_mean = np.load('/home/lei/dataset_all/Mean_pos.npy')
        self.pos_std = np.load('/home/lei/dataset_all/Std_pos.npy')

        self.training_control_joint = np.array(model_params.training_control_joint)
        self.testing_control_joint = np.array(model_params.testing_control_joint)

        self.training_density = model_params.training_density
        self.testing_density = model_params.testing_density

        self.data_dict = data_dict
        self.nfeats = data_dict[name_list[0]]["motion"].shape[1]
        self.name_list = name_list

        for key in self.data_dict:
            keykey = key
            o_key = key
            if "_" in key:
                key = key[2:]

            flag = 1
            for _key in graph:
                if o_key == _key:
                    
                    self.data_dict[o_key]["text"] = graph[_key]
                    flag = 0
                    break
            
            if flag:
                for _key in graph:
                    o__key = _key
                    if "_" in _key:
                        _key = _key[2:]
                    if key == _key:
                        
                        self.data_dict[o_key]["text"] = graph[o__key]
                        break

    def __len__(self) -> int:
        return len(self.name_list)

    def random_mask(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        choose_joint = self.testing_control_joint

        length = joints.shape[0]
        density = self.testing_density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints #* mask_seq
        return joints, mask_seq

    def random_mask_train(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        num_joints = len(self.training_control_joint)
        num_joints_control = 1
        choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
        choose_joint = self.training_control_joint[choose_joint]

        length = joints.shape[0]

        if self.training_density == 'random':
            choose_seq_num = np.random.choice(length - 1, 1) + 1
        else:
            choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()
        
        choose_pose_num = 1
        choose_pose_idx = int(random.sample(sorted(choose_seq), choose_pose_num)[0])
        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints #* mask_seq
        return joints, mask_seq

    def __getitem__(self, idx: int) -> tuple:
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list,pose = data["motion"], data["length"], data["text"], data["pose"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        V = []
        motion_idx = []

        motion_len = len(pose)
        is_first = False
        selected_frames = 0
        if "V" in text_data.keys():
            for i in text_data["V"]:
                V.append(text_data["V"][i]['spans'])
                motion_list = text_data["V"][i]['motion_idx']
                
                if self.mode == 'train':
                    motion_item = random.choice(motion_list)
                else:
                    motion_item = motion_list[0]

                if is_first == False:
                    selected_frames = motion_item
                    is_first = True
                    
                motion_idx.append(motion_item)
        else:
            if self.mode == 'train':
                motion_item = random.randint(0,motion_len-1)
            else:
                motion_item = 0

            if is_first == False:
                selected_frames = motion_item
                is_first = True
            
            motion_idx.append(motion_item)

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)

        motion = motion[idx:idx + m_length]
        pose = pose[idx:idx + m_length]

        hint = None
        if self.mode is not None:
            n_joints = 22 if motion.shape[-1] == 263 else 21
            # hint is global position of the controllable joints
            joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
            joints = joints.numpy()

            # control any joints at any time
            if self.mode == 'train':
                hint,mask_seq = self.random_mask_train(joints, n_joints)
            else:
                hint,mask_seq = self.random_mask(joints, n_joints)

            # hint = hint.reshape(hint.shape[0], -1)
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        pose = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        T = pose.shape[0]  # 帧数
        num_frames_to_select = 1

        selected_frames = min(selected_frames,T-1)
        selected_frames = max(0,selected_frames)

        mask = np.zeros(T, dtype=int)

        mask[selected_frames] = 1
        mask_seq[selected_frames] = 1

        mask = mask[:, np.newaxis]
        pose = pose.reshape(-1,n_joints*3)
        pose = pose * mask

        hint = hint * mask_seq
        hint = hint.reshape(hint.shape[0], -1)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            pose,
            mask,
            hint
        )

class Text2MotionDatasetRandomKeyPoseTrejControl(data.Dataset):

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        w_vectorizer: WordVectorizer,
        max_motion_length: int,
        min_motion_length: int,
        max_text_len: int,
        unit_length: int,
        motion_dir: str,
        text_dir: str,
        fps: int,
        tiny: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        maxdata = 10 if tiny else 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []

        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if len(motion) < self.min_motion_length or len(motion) >= self.max_motion_length:
                    bad_count += 1
                    continue
                text_data = []
                flag = False

                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < self.min_motion_length or \
                                        len(n_motion) >= self.max_motion_length:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if not tiny:
            logger.info(f"Reading {len(self.id_list)} motions from {split_file}.")
            logger.info(f"Total {len(name_list)} motions are used.")
            logger.info(f"{bad_count} motion sequences not within the length range of "
                        f"[{self.min_motion_length}, {self.max_motion_length}) are filtered out.")

        self.mean = mean
        self.std = std

        self.mode = None
        model_params = kwargs['model_kwargs']
        if 'is_controlnet' in model_params and model_params.is_controlnet is True:
            if 'test' in split_file or 'val' in split_file:
                self.mode = 'eval'
            else:
                self.mode = 'train'

            self.t_ctrl = model_params.is_controlnet_temporal
            spatial_norm_path = './datasets/humanml_spatial_norm'
            self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
            self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

            self.training_control_joint = np.array(model_params.training_control_joint)
            self.testing_control_joint = np.array(model_params.testing_control_joint)

            self.training_density = model_params.training_density
            self.testing_density = model_params.testing_density

        self.data_dict = data_dict
        self.nfeats = data_dict[name_list[0]]["motion"].shape[1]
        self.name_list = name_list

    def __len__(self) -> int:
        return len(self.name_list)

    def random_mask(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        choose_joint = self.testing_control_joint

        length = joints.shape[0]
        density = self.testing_density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints #* mask_seq
        return joints , mask_seq

    def random_mask_train(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        if self.t_ctrl:
            choose_joint = self.training_control_joint
        else:
            num_joints = len(self.training_control_joint)
            num_joints_control = 1
            choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
            choose_joint = self.training_control_joint[choose_joint]

        length = joints.shape[0]

        if self.training_density == 'random':
            choose_seq_num = np.random.choice(length - 1, 1) + 1
        else:
            choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints #* mask_seq
        return joints,mask_seq

    def __getitem__(self, idx: int) -> tuple:
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        hint = None
        if self.mode is not None:
            n_joints = 22 if motion.shape[-1] == 263 else 21
            # hint is global position of the controllable joints
            joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
            joints = joints.numpy()

            # control any joints at any time
            if self.mode == 'train':
                hint,mask_seq = self.random_mask_train(joints, n_joints)
            else:
                hint,mask_seq = self.random_mask(joints, n_joints)

            # hint = hint.reshape(hint.shape[0], -1)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        pose = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        T = pose.shape[0]  # 帧数
        num_frames_to_select = 1 

        selected_frames = random.sample(range(T), num_frames_to_select)  
        mask = np.zeros(T, dtype=int)

        mask[selected_frames] = 1

        mask = mask[:, np.newaxis]
        pose = pose.reshape(-1,n_joints*3)
        pose = pose * mask


        hint = hint * mask_seq
        hint = hint.reshape(hint.shape[0], -1)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            pose,
            mask,
            hint
        )