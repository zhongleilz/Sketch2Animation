import time
import inspect
import logging
from typing import Optional

import tqdm
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor, sum_flat,count_trainable_parameters
from .base import BaseModel
import torch.optim as optim
from mld.models.architectures.pose_encoder import PoseEncoder
from mld.data.humanml.scripts.motion_process import recover_from_ric,extract_rotations

logger = logging.getLogger(__name__)


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        x = x.permute(1,0,2).squeeze(1)
        y = y.permute(1,0,2).squeeze(1)

        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"


class MLD(BaseModel):
    def __init__(self, cfg: DictConfig, datamodule: BaseDataModule) -> None:
        super().__init__()

        self.cfg = cfg
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncondp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        self.is_keypose_2d = cfg.model.is_keypose_2d

        cfg_denoiser = self.cfg.model.denoiser.copy()
        cfg_denoiser['params']["is_keypose_2d"] = self.is_keypose_2d 

        self.denoiser = instantiate_from_config(cfg_denoiser)
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(cfg.model.noise_scheduler)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints
        self.norm = datamodule.norm
        self.feats2rotations = datamodule.feats2rotations

        self.alphas = torch.sqrt(self.noise_scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.noise_scheduler.alphas_cumprod)
        self.l2_loss = lambda a, b: (a - b) ** 2

        self.is_controlnet = cfg.model.is_controlnet
        self.is_traj_grad = cfg.model.is_traj_grad
        self.is_traj_2d = cfg.model.is_traj_2d

        self.is_keypose_grad = cfg.model.is_keypose_grad
        self.is_keypose_2d = cfg.model.is_keypose_2d

        self.raw_mean = np.load('./datasets/humanml_spatial_norm/Mean_raw.npy')
        self.raw_std = np.load('./datasets/humanml_spatial_norm/Std_raw.npy')

        self.raw_mean = torch.from_numpy(self.raw_mean).cuda()
        self.raw_std = torch.from_numpy(self.raw_std).cuda()

        self.mean_2d = torch.from_numpy(np.load('./datasets/humanml3d/Mean_pos_2d.npy')).cuda()
        self.std_2d = torch.from_numpy(np.load('./datasets/humanml3d/Std_pos_2d.npy')).cuda()

        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        if self.is_controlnet:
            c_cfg = self.cfg.model.denoiser.copy()
            c_cfg['params']['is_controlnet'] = True
            self.controlnet = instantiate_from_config(cfg.model.controlnet)
            self.training_control_joint = cfg.model.training_control_joint
            self.testing_control_joint = cfg.model.testing_control_joint
            self.training_density = cfg.model.training_density
            self.testing_density = cfg.model.testing_density
            self.control_scale = cfg.model.control_scale
            self.vaeloss = cfg.model.vaeloss
            self.vaeloss_type = cfg.model.vaeloss_type
            self.cond_ratio = cfg.model.cond_ratio
            self.rot_ratio = cfg.model.rot_ratio

            self.is_controlnet_temporal = cfg.model.is_controlnet_temporal
            self.traj_encoder = instantiate_from_config(cfg.model.traj_encoder)
            if self.is_traj_2d:
                self.traj_encoder_2d = instantiate_from_config(cfg.model.traj_encoder)                
                self.contrastive_loss_fn = InfoNCE_with_filtering()

            logger.info(f"control_scale: {self.control_scale}, vaeloss: {self.vaeloss}, "
                        f"cond_ratio: {self.cond_ratio}, rot_ratio: {self.rot_ratio}, "
                        f"vaeloss_type: {self.vaeloss_type}")
            logger.info(f"is_controlnet_temporal: {self.is_controlnet_temporal}")
            logger.info(f"training_control_joint: {self.training_control_joint}")
            logger.info(f"testing_control_joint: {self.testing_control_joint}")
            logger.info(f"training_density: {self.training_density}")
            logger.info(f"testing_density: {self.testing_density}")

            time.sleep(2)

        self.summarize_parameters()

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1 and self.denoiser.time_cond_proj_dim is None

    def summarize_parameters(self) -> None:
        logger.info(f'VAE Encoder: {count_trainable_parameters(self.vae.encoder)}M')
        logger.info(f'VAE Decoder: {count_trainable_parameters(self.vae.decoder)}M')
        logger.info(f'Denoiser: {count_trainable_parameters(self.denoiser)}M')

        if self.is_controlnet:
            traj_encoder = count_trainable_parameters(self.traj_encoder)
            controlnet = count_trainable_parameters(self.controlnet)
            logger.info(f'ControlNet: {controlnet}M')
            logger.info(f'traj_encoder: {traj_encoder}M')


    def forward(self, batch: dict) -> tuple:
        texts = batch["text"]
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            texts = [""] * len(texts) + texts

        text_emb = self.text_encoder(texts)

        hint = batch['hint'] if 'hint' in batch else None  # control signals
        z = self._diffusion_reverse(text_emb, hint)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if 'motion' in batch:
            feats_ref = batch['motion']
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref
        

    def forward_gmld_sequence_2d_keypose_traj(self, batch: dict) -> tuple:
        texts = batch["text"]
        lengths = batch["length"]

        pose = batch["pose"]
        pose_2d = batch["pose_2d"]
        pose_mask = batch["pose_mask"].bool()
        motion_idx = batch["motion_idx"]

        rotation = batch["rotation"]

        bsz = len(texts)
        if self.do_classifier_free_guidance:
          
            texts = [""] * len(texts) + texts

            pose = pose * pose_mask
            pose_2d = pose_2d * pose_mask

        cond_emb = self.text_encoder(texts)

        hint = batch['hint'] if 'hint' in batch else None  # control signals
        hint_2d = batch['hint_2d']

        z = self._diffusion_control_reverse_2d_keypose(cond_emb,lengths, hint_2d = hint_2d, rotation = rotation, pose_2d = pose_2d, pose_2d_mask = pose_mask)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if 'motion' in batch:
            feats_ref = batch['motion']
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref


    def predicted_origin(self, model_output, timesteps, sample):
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
        return pred_x_0

    
    def _diffusion_control_reverse_2d_keypose(self, encoder_hidden_states: torch.Tensor,lengths: torch.Tensor, hint: torch.Tensor = None,hint_2d: torch.Tensor = None,pose_2d: torch.Tensor = None, pose_2d_mask: torch.Tensor = None, rotation: torch.Tensor = None) -> torch.Tensor:

        controlnet_cond = None
        if self.is_controlnet and (hint is not None or hint_2d is not None):
            if hint is not None:
                hint_mask = hint.sum(-1) != 0
                controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
                controlnet_cond = controlnet_cond.permute(1, 0, 2)
            else:
                hint_2d_mask = hint_2d.sum(-1) != 0
                controlnet_cond = self.traj_encoder_2d(hint_2d, mask=hint_2d_mask)
                controlnet_cond = controlnet_cond.permute(1, 0, 2)

        # init latents
        bsz = encoder_hidden_states.shape[0]
        
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        
        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states.device,
            dtype=torch.float)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
            
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # reverse
        for i, t in tqdm.tqdm(enumerate(timesteps)):
            current_t = int(t.cpu().numpy())-1
            # expand the latents if we are doing classifier free guidance

            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            controlnet_residuals = None
            if self.is_controlnet and controlnet_cond is not None:
                if self.do_classifier_free_guidance:
                    controlnet_prompt_embeds = encoder_hidden_states.chunk(2)[1]
                else:
                    controlnet_prompt_embeds = encoder_hidden_states

                controlnet_residuals = self.controlnet(
                    latents,
                    t,
                    timestep_cond=timestep_cond,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond)

                if self.do_classifier_free_guidance:
                    controlnet_residuals = [torch.cat([torch.zeros_like(d), d * self.control_scale], dim=1)
                                            for d in controlnet_residuals]
                else:
                    controlnet_residuals = [d * self.control_scale for d in controlnet_residuals]


            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                pose_cond=pose_2d,
                pose_mask=pose_2d_mask,
                controlnet_residuals=controlnet_residuals
                )

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)   

            
            if self.is_traj_grad and self.is_traj_2d == True:
                noise_pred = self.guide_traj_bfgs_2d(noise_pred,latents,lengths,current_t,hint=hint_2d, rotation=rotation)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents


    def guide_traj(self, noise_pred,latents,lengths, timestep, hint, t_stopgrad=-10, n_scale=0.05, n_guide_steps=20, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        n_joint = 22

        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        if timestep > 500:
            n_guide_steps = 50
        else:
            n_guide_steps = 20

        # process hint
        mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        if self.raw_std.device != hint.device:
            self.raw_mean = self.raw_mean.to(hint.device)
            self.raw_std = self.raw_std.to(hint.device)
          
        hint = hint * self.raw_std + self.raw_mean
        hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint
        # joint id
        joint_ids = []
        for m in mask_hint:
            joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
            joint_ids.append(joint_id)
        
        scale = float(self.calc_grad_scale(mask_hint)[0][0][0])

        for _ in range(n_guide_steps):
            loss, grad = self.gradients_traj(noise_pred,latents,timestep, hint, mask_hint,lengths, joint_ids)
            # grad = model_variance * grad
            
            if timestep >= t_stopgrad:
                
                noise_pred = noise_pred - scale * grad * n_scale
        
        # print("-------------------")
        return noise_pred.detach()

    def guide_traj_2d(self, noise_pred,latents,lengths, timestep, hint=None, rotation = None, t_stopgrad=-10, n_scale=0.05, n_guide_steps=20, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        n_joint = 22

        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        if timestep > 500:
            n_guide_steps = 10
        else:
            n_guide_steps = 10

        # process hint
        mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        if self.raw_std.device != hint.device:
            self.raw_mean = self.raw_mean.to(hint.device)
            self.raw_std = self.raw_std.to(hint.device)
          
        hint = hint * self.raw_std + self.raw_mean
        hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint
        # joint id
        joint_ids = []
        for m in mask_hint:
            joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
            joint_ids.append(joint_id)
        
        scale = float(self.calc_grad_scale(mask_hint)[0][0][0])

        for _ in range(n_guide_steps):
            loss, grad = self.gradients_traj_2d(noise_pred,latents,timestep, hint, mask_hint,lengths,rotation, joint_ids)
            # grad = model_variance * grad
            # print("loss:::",loss.sum())
            
            if timestep >= t_stopgrad:
                
                noise_pred = noise_pred - scale * grad * n_scale
        
        # print("-------------------")
        return noise_pred.detach()

    def guide_traj_bfgs(self, noise_pred,latents,lengths, timestep, hint, t_stopgrad=-10, n_scale=0.05, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        n_joint = 22

        # prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        if train:
            n_guide_steps = 1
        else:
            if timestep > 500:
                n_guide_steps = 1
            else:
                n_guide_steps = 2

        # process hint
        mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        if self.raw_std.device != hint.device:
            self.raw_mean = self.raw_mean.to(hint.device)
            self.raw_std = self.raw_std.to(hint.device)
          
        hint = hint * self.raw_std + self.raw_mean
        hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint
        # joint id
        joint_ids = []
        for m in mask_hint:
            joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
            joint_ids.append(joint_id)
        
        scale = float(self.calc_grad_scale(mask_hint)[0][0][0])

        with torch.enable_grad():
            noise_pred = noise_pred.clone().detach().contiguous().requires_grad_(True)

            def closure():
                lbfgs.zero_grad()
                objective = self.bfgs_traj(noise_pred,latents,timestep, hint, mask_hint,lengths, joint_ids)
                
                objective.backward()
                return objective

            lbfgs = optim.LBFGS([noise_pred],
                    history_size=10, 
                    max_iter=4, 
                    line_search_fn="strong_wolfe")
            for _ in range(n_guide_steps):
                lbfgs.step(closure)
        

        return noise_pred


    def guide_traj_bfgs_2d(self, noise_pred,latents,lengths, timestep, hint=None, rotation = None, pose_2d=None, t_stopgrad=-10, n_scale=0.05, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        n_joint = 22

        if train:
            n_guide_steps = 1
        else:
            if timestep > 500:
                n_guide_steps = 1
            else:
                n_guide_steps = 2

        # process hint
        if hint is not None:
            mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        
        if pose_2d is not None:
            mask_pose_2d = pose_2d.view(pose_2d.shape[0], pose_2d.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0


        if self.raw_std.device != noise_pred.device:
            self.raw_mean = self.raw_mean.to(noise_pred.device)
            self.raw_std = self.raw_std.to(noise_pred.device)
        
        if hint is not None:
            hint = hint * self.raw_std + self.raw_mean
            hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint
        
        if pose_2d is not None:
            pose_2d = pose_2d * self.raw_std + self.raw_mean
            pose_2d = pose_2d.view(pose_2d.shape[0], pose_2d.shape[1], n_joint, 3) * mask_pose_2d

        # joint id
        joint_ids = []
        
        if hint is not None:
            for m in mask_hint:
                joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
                joint_ids.append(joint_id)

        if pose_2d is not None:
            for m in mask_pose_2d:
                joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
                joint_ids.append(joint_id)    
        
        with torch.enable_grad():
            noise_pred = noise_pred.clone().detach().contiguous().requires_grad_(True)

            def closure():
                lbfgs.zero_grad()
                if hint is not None:
                    objective = self.bfgs_traj_2d(noise_pred, latents, timestep, hint, mask_hint, lengths, rotation, joint_ids)
                
                if pose_2d is not None:
            
                    objective = self.bfgs_traj_2d(noise_pred, latents, timestep, pose_2d, mask_pose_2d, lengths, rotation, joint_ids)
                
                objective.backward()
                return objective

            lbfgs = optim.LBFGS([noise_pred],
                    history_size=10, 
                    max_iter=8, 
                    line_search_fn="strong_wolfe")
            for _ in range(n_guide_steps):
                lbfgs.step(closure)
        

        return noise_pred

    def calc_grad_scale(self, mask_hint):
        # assert mask_hint.shape[1] == 196
        num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        scale = 20 / max_keyframes
        return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def get_clean_sample(self,noise_pred,latents,timestep):
        # prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        if not isinstance(timestep, int):
            timestep = timestep.to(self.scheduler.alphas_cumprod.device)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        if not isinstance(timestep, int):
            alpha_prod_t = alpha_prod_t.view(-1, 1, 1).cuda()
            beta_prod_t = beta_prod_t.view(-1, 1, 1).cuda()

        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
    
        return pred_original_sample


    def gradients_traj(self, noise_pred,latents,timestep, hint, mask_hint, lengths,joint_ids=None):
        with torch.enable_grad():
            noise_pred.requires_grad_(True)
            batch_size = noise_pred.shape[0]

            clean_sample = self.get_clean_sample(noise_pred,latents,timestep)
            feats = self.vae.decode(clean_sample.permute(1, 0, 2).contiguous(), lengths)
            joint_pos = self.feats2joints(feats)#.permute(1,0,2,3)
        
            hint = hint.cuda()[0:1,:,:,:]
            mask_hint = mask_hint.cuda()[0:1,:,:,:]
          
            loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
            grad = torch.autograd.grad([loss.sum()], [noise_pred])[0]
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            grad[..., 0] = 0
            noise_pred.detach()
        return loss, grad

    
    def gradients_traj_2d(self, noise_pred,latents,timestep, hint, mask_hint, lengths,rotation,joint_ids=None):
        with torch.enable_grad():
            noise_pred.requires_grad_(True)
            batch_size = noise_pred.shape[0]

            clean_sample = self.get_clean_sample(noise_pred,latents,timestep)
            feats = self.vae.decode(clean_sample.permute(1, 0, 2).contiguous(), lengths)
            joint_pos = self.feats2joints(feats)#.permute(1,0,2,3)

            if joint_pos.shape[0] != rotation.shape[0]:
                rotation = rotation.repeat(joint_pos.shape[0], 1, 1)

            joint_pos_2d = self.project_to_2d(joint_pos,rotation)
            joint_pos_2d = joint_pos_2d.view(joint_pos_2d.shape[0], joint_pos_2d.shape[1], self.njoints, 3)[...,:2]

            hint = hint.cuda()[...,:2]
            mask_hint = mask_hint.cuda()[...,:2]

            _,len_hint,_,_ = hint.shape
            _,len_joint,_,_ = joint_pos_2d.shape
            min_len = min(len_hint,len_joint)
          
            loss = torch.norm((joint_pos_2d[:,:min_len] - hint[:,:min_len]) * mask_hint[:,:min_len], dim=-1)
            grad = torch.autograd.grad([loss.sum()], [noise_pred])[0]
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            grad[..., 0] = 0
            noise_pred.detach()
        return loss, grad


    def bfgs_traj(self, noise_pred,latents,timestep, hint, mask_hint, lengths,joint_ids=None):
        with torch.enable_grad():
            noise_pred.requires_grad_(True)
            batch_size = noise_pred.shape[0]

            clean_sample = self.get_clean_sample(noise_pred,latents,timestep)
            feats = self.vae.decode(clean_sample.permute(1, 0, 2).contiguous(), lengths)
            joint_pos = self.feats2joints(feats)#.permute(1,0,2,3)
            hint = hint.cuda()
            mask_hint = mask_hint.cuda()

            

            loss = self.mse_loss(joint_pos*mask_hint, hint*mask_hint)
        return loss

    
    def bfgs_traj_2d(self, noise_pred,latents,timestep, hint, mask_hint, lengths,rotation, joint_ids=None):
        with torch.enable_grad():
            noise_pred.requires_grad_(True)
            batch_size = noise_pred.shape[0]

            clean_sample = self.get_clean_sample(noise_pred,latents,timestep)
            feats = self.vae.decode(clean_sample.permute(1, 0, 2).contiguous(), lengths)
            joint_pos = self.feats2joints(feats)#.permute(1,0,2,3)

            if joint_pos.shape[0] != rotation.shape[0]:
                rotation = rotation.repeat(joint_pos.shape[0], 1, 1)

            joint_pos_2d = self.project_to_2d(joint_pos,rotation)
            joint_pos_2d = joint_pos_2d.view(joint_pos_2d.shape[0], joint_pos_2d.shape[1], self.njoints, 3)[...,:2]

            hint = hint.cuda()[...,:2]
            mask_hint = mask_hint.cuda()[...,:2]

            _,len_hint,_,_ = hint.shape
            _,len_joint,_,_ = joint_pos_2d.shape
            min_len = min(len_hint,len_joint)

            loss = self.mse_loss(joint_pos_2d[:,:min_len] * mask_hint[:,:min_len], hint[:,:min_len] * mask_hint[:,:min_len])
            
        return loss

    
    def project_to_2d(self, motion_3d, rotation_matrix):
       
        B, F, J, _ = motion_3d.shape
        
        motion_3d_flat = motion_3d.view(B, -1, 3)  # [B, F*J, 3]
        
        motion_3d_rotated = torch.bmm(motion_3d_flat, rotation_matrix.transpose(1, 2))  # [B, F*J, 3]
                
        motion_2d = motion_3d_rotated.view(B, F, -1)  # [B, F, J, 2]
        
        return motion_2d

    def t2m_eval(self, batch: dict) -> dict:
        texts = batch["text"]
        feats_ref = batch["motion"]
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        start = time.time()

        if self.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.do_classifier_free_guidance:
            texts = [""] * len(texts) + texts

        text_st = time.time()
        text_emb = self.text_encoder(texts)
        text_et = time.time()
        self.text_encoder_times.append(text_et - text_st)

        diff_st = time.time()
        hint = batch['hint'] if 'hint' in batch else None  # control signals
        z = self._diffusion_reverse(text_emb, hint)
        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if 'motion' in batch:
            feats_ref = batch['motion']
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

