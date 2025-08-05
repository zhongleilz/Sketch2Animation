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

        self.raw_mean = np.load('/home/lei/MotionLCM2/datasets/humanml_spatial_norm/Mean_raw.npy')
        self.raw_std = np.load('/home/lei/MotionLCM2/datasets/humanml_spatial_norm/Std_raw.npy')

        self.raw_mean = torch.from_numpy(self.raw_mean).cuda()
        self.raw_std = torch.from_numpy(self.raw_std).cuda()

        self.mean_2d = torch.from_numpy(np.load('/home/lei/dataset_all/Mean_pos_2d.npy')).cuda()
        self.std_2d = torch.from_numpy(np.load('/home/lei/dataset_all/Std_pos_2d.npy')).cuda()

        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        if self.is_controlnet:
            c_cfg = self.cfg.model.denoiser.copy()
            c_cfg['params']['is_controlnet'] = True
            # self.controlnet = instantiate_from_config(c_cfg)
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
                # noise_pred = self.guide_traj_2d(noise_pred,latents,lengths,current_t,hint=hint_2d, rotation=rotation)
                noise_pred = self.guide_traj_bfgs_2d(noise_pred,latents,lengths,current_t,hint=hint_2d, rotation=rotation)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents


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
                    max_iter=8, 
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
        
        
        # scale = float(self.calc_grad_scale(mask_hint)[0][0][0])

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
                    max_iter=2, 
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
        if not isinstance(timestep, int):
            timestep = timestep.to(self.scheduler.alphas_cumprod.device)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        if not isinstance(timestep, int):
            alpha_prod_t = alpha_prod_t.view(-1, 1, 1).cuda()
            beta_prod_t = beta_prod_t.view(-1, 1, 1).cuda()

        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
    
        return pred_original_sample


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

    def _diffusion_reverse(self, encoder_hidden_states: torch.Tensor, hint: torch.Tensor = None) -> torch.Tensor:

        controlnet_cond = None
        if self.is_controlnet and hint is not None:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
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
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            controlnet_residuals = None
            if self.is_controlnet:
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
                controlnet_residuals=controlnet_residuals)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents
    

    def _diffusion_control_process(self, latents, encoder_hidden_states, lengths=None,pose=None,pose_mask=None,hint=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]

        controlnet_cond = None
        if self.is_controlnet and hint is not None:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
            controlnet_cond = controlnet_cond.permute(1, 0, 2)

        latents = latents.permute(1, 0, 2)

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        
        # current_t = int(timesteps.cpu().numpy())-1

        controlnet_residuals = None
        if self.is_controlnet and controlnet_cond is not None:
            controlnet_residuals = self.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond)

        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=timestep_cond,
            encoder_hidden_states=encoder_hidden_states,
            pose_cond=pose,
            pose_mask=pose_mask,
            controlnet_residuals=controlnet_residuals
        )
 
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.is_traj_grad:
            noise_pred = self.guide_traj_bfgs(noise_pred,noisy_latents,lengths, timesteps, hint, train = True)

        model_pred = self.predicted_origin(noise_pred, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "model_pred": model_pred,
            "model_gt": latents
        }
        return n_set


    def _diffusion_lcm_control_process(self, latents, encoder_hidden_states, lengths=None, pose=None, pose_mask=None, hint=None):
        """
        Modified for multi-step training (num_inference_timesteps > 1).
        """
        controlnet_cond = None
        if self.is_controlnet and hint is not None:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
            controlnet_cond = controlnet_cond.permute(1, 0, 2)

        latents = latents.permute(1, 0, 2)

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample multiple timesteps for multi-step training
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, self.scheduler.num_inference_steps),  # num_inference_timesteps > 1
            device=latents.device,
        )
        timesteps = timesteps.sort(dim=1, descending=True)[0]  # Ensure descending order
        noisy_latents = latents.clone()

        total_loss = 0
        predictions = []

        for i in range(self.scheduler.num_inference_steps):
            # Add noise for this timestep
            noisy_latents = self.noise_scheduler.add_noise(noisy_latents, noise, timesteps[:, i])

            # Compute ControlNet residuals
            controlnet_residuals = None
            if self.is_controlnet and controlnet_cond is not None:
                controlnet_residuals = self.controlnet(
                    sample=noisy_latents,
                    timestep=timesteps[:, i],
                    timestep_cond=timestep_cond,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                )

            # Predict noise residual
            noise_pred = self.denoiser(
                sample=noisy_latents,
                timestep=timesteps[:, i],
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                pose_cond=pose,
                pose_mask=pose_mask,
                controlnet_residuals=controlnet_residuals,
            )

            if self.is_traj_grad:
                noise_pred = self.guide_traj_bfgs(noise_pred,noisy_latents,lengths, timesteps[:, i], hint, train = True)

            # Predicted latent
            model_pred = self.predicted_origin(noise_pred, timesteps[:, i], noisy_latents)
            predictions.append(model_pred)

            # Update noisy_latents for the next step
            noisy_latents = model_pred.clone()

        # Collect outputs
        n_set = {
            "noise": noise,
            "model_preds": predictions,
            "model_gt": latents,
        }
        return n_set



    def _diffusion_control_process_2d(self, latents, encoder_hidden_states, lengths=None,pose=None,pose_mask=None,hint_2d=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]

        controlnet_cond = None
        if self.is_controlnet and hint_2d is not None:
            hint_2d_mask = hint_2d.sum(-1) != 0
            # pose_mask = pose.sum(-1) != 0
            controlnet_cond = self.traj_encoder_2d(hint_2d, mask=hint_2d_mask)
            # pose_cond = self.pose_encoder_2d(pose,mask=pose_mask)

            controlnet_cond = controlnet_cond #+ pose_cond
            controlnet_cond = controlnet_cond.permute(1, 0, 2)

        latents = latents.permute(1, 0, 2)

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        
        # current_t = int(timesteps.cpu().numpy())-1

        controlnet_residuals = None
        if self.is_controlnet and controlnet_cond is not None:
            controlnet_residuals = self.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond)

        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=timestep_cond,
            encoder_hidden_states=encoder_hidden_states,
            pose_cond=pose,
            pose_mask=pose_mask,
            controlnet_residuals=controlnet_residuals
        )
 
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        # if self.is_traj_grad:
        #     noise_pred = self.guide_traj_bfgs(noise_pred,noisy_latents,lengths, timesteps, hint, train = True)
        if self.is_traj_grad and self.is_traj_2d == True:
                noise_pred = self.guide_traj_bfgs_2d(noise_pred,latents,lengths,current_t, hint_2d, rotation)

        model_pred = self.predicted_origin(noise_pred, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "model_pred": model_pred,
            "model_gt": latents
        }
        return n_set


    def _diffusion_process(self, latents: torch.Tensor, encoder_hidden_states: torch.Tensor,
                           hint: torch.Tensor = None) -> dict:

        controlnet_cond = None
        if self.is_controlnet:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
            controlnet_cond = controlnet_cond.permute(1, 0, 2)

        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise, timesteps)

        controlnet_residuals = None
        if self.is_controlnet:
            controlnet_residuals = self.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond)

        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=timestep_cond,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_residuals=controlnet_residuals)

        model_pred = self.predicted_origin(noise_pred, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "model_pred": model_pred,
            "model_gt": latents
        }
        return n_set

    def masked_l2(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())
        n_entries = a.shape[-1]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss = loss / non_zero_elements
        return mse_loss.mean()


    def train_control_lcm_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # pose = batch["pose"]
        # pose_mask = batch["pose_mask"]
        hint = batch['hint'] if 'hint' in batch else None 

        # Encode motion
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)

        # Classifier-free guidance: randomly drop text
        text = batch["text"]
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]

        # Encode text
        cond_emb = self.text_encoder(text)

        # Perform diffusion process
        # n_set = self._diffusion_lcm_control_process(z, cond_emb, lengths, pose=pose, pose_mask=pose_mask, hint=hint)
        n_set = self._diffusion_lcm_control_process(z, cond_emb, lengths, hint=hint)

        # Loss calculation
        loss_dict = dict()
        is_pose_loss = False
        is_drop_hint = True
        model_preds = n_set['model_preds']
        target = n_set['model_gt']

        # Diffusion loss (average over timesteps)
        diff_loss = 0
        for model_pred in model_preds:
            diff_loss += F.mse_loss(model_pred, target, reduction="mean")
        diff_loss /= self.scheduler.num_inference_steps
        loss_dict['diff_loss'] = diff_loss

        # Keypose loss
        if self.is_controlnet and is_pose_loss:
            z_pred = model_preds[-1]  # Use the final step's prediction for keypose loss
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)

            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)

            pose_temp = pose.view(pose.shape[0], pose.shape[1], self.njoints, 3)
            mask_pose = pose_temp.sum(dim=-1, keepdim=True) != 0
            pose_mask = pose_mask.unsqueeze(-1)

            if self.cond_ratio != 0:
                cond_loss = (self.l2_loss(joints_rst, pose_temp).sum(-1, keepdims=True) * mask_pose).sum() / mask_pose.sum()
                loss_dict['keypose_loss'] = 2 * self.cond_ratio * cond_loss
            else:
                loss_dict['keypose_loss'] = torch.tensor(0., device=diff_loss.device)
        else:
            loss_dict['keypose_loss'] = torch.tensor(0., device=diff_loss.device)

        # Trajectory loss
        if self.is_controlnet and is_drop_hint == False:
            z_pred = model_preds[-1]  # Use the final step's prediction for traj loss
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)

            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)

            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3)
            mask_hint = hint.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                cond_loss = (self.l2_loss(joints_rst, hint).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                loss_dict['traj_loss'] = self.cond_ratio * cond_loss
            else:
                loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)
        else:
            loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)

        # Total loss
        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict

    def train_control_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # pose = batch["pose"]
        # pose_mask = batch["pose_mask"]
        hint = batch['hint'] if 'hint' in batch else None 

        # motion encode
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)


        text = batch["text"]
        # classifier free guidance: randomly drop text during training
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        # text encode
        cond_emb = self.text_encoder(text)
        
        # n_set = self._diffusion_control_process(z, cond_emb, lengths,pose=pose,pose_mask=pose_mask,hint=hint)
        n_set = self._diffusion_control_process(z, cond_emb, lengths,hint=hint)
      
        loss_dict = dict()
        is_pose_loss = False
        is_drop_hint = False

        if self.denoiser.time_cond_proj_dim is not None:
            # LCM
            model_pred = n_set['model_pred']
            target = n_set['model_gt']
            # Performance comparison: l2 loss > huber loss
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            model_pred = n_set['noise_pred']
            target = n_set['noise']
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss
        loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)
        
        if self.is_controlnet and is_pose_loss:
            z_pred = n_set['model_pred']
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)

            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1) 
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)

            pose_temp = batch['pose']
            pose_temp = pose_temp.view(pose_temp.shape[0], pose_temp.shape[1], self.njoints, 3)
            mask_pose = pose_temp.sum(dim=-1, keepdim=True) != 0
            pose_mask = pose_mask.unsqueeze(-1)


            if self.cond_ratio != 0:
                cond_loss = (self.l2_loss(joints_rst, pose_temp).sum(-1, keepdims=True) * mask_pose).sum() / mask_pose.sum()
                loss_dict['keypose_loss'] =  2 * self.cond_ratio * cond_loss
            else:
                loss_dict['keypose_loss'] = torch.tensor(0., device=diff_loss.device)
        else:
                loss_dict['keypose_loss'] = torch.tensor(0., device=diff_loss.device)


        if self.is_controlnet and is_drop_hint == False:
            z_pred = n_set['model_pred']
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)

            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)            

            hint = batch['hint']
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3)
            mask_hint = hint.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                cond_loss = (self.l2_loss(joints_rst, hint).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                loss_dict['traj_loss'] = self.cond_ratio * cond_loss 


            else:
                loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)
        else:
            loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)

        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict

    def info_nce_loss(self, controlnet_cond, controlnet_cond_2d, tau=0.07):
        """
        基于 InfoNCE 的损失函数，适配输入 shape [batch_size, 1, embedding_dim]
        :param controlnet_cond: 3D trajectory embeddings, [batch_size, 1, embedding_dim]
        :param controlnet_cond_2d: 2D trajectory embeddings, [batch_size, 1, embedding_dim]
        :param tau: 温度参数
        :return: InfoNCE 损失值
        """
        # 输入检查
        assert controlnet_cond.dim() == 3 and controlnet_cond_2d.dim() == 3, \
            "Expected input to have shape [batch_size, 1, embedding_dim]"

        controlnet_cond = controlnet_cond.permute(1,0,2)
        controlnet_cond_2d = controlnet_cond_2d.permute(1,0,2)

        batch_size, _, embedding_dim = controlnet_cond.size()

        # 将 controlnet_cond 和 controlnet_cond_2d 展平到 [batch_size, embedding_dim]
        controlnet_cond = controlnet_cond.squeeze(1)  # [batch_size, embedding_dim]
        controlnet_cond_2d = controlnet_cond_2d.squeeze(1)  # [batch_size, embedding_dim]

        controlnet_cond = F.normalize(controlnet_cond, p=2, dim=1)  # [batch_size, embedding_dim]
        controlnet_cond_2d = F.normalize(controlnet_cond_2d, p=2, dim=1)

        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(
            controlnet_cond, controlnet_cond_2d.T
        )  # [batch_size, batch_size]

        # 应用温度缩放
        similarity_matrix /= tau

        # 构造标签：对角线元素是正样本
        labels = torch.arange(batch_size).to(similarity_matrix.device)  # [batch_size]

        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
    
    def train_control_diffusion_traj2d_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # pose = batch["pose"]
        # pose_mask = batch["pose_mask"]
        hint = batch['hint'] if 'hint' in batch else None 
        hint_2d_a = batch['hint_2d_a'] 
        hint_2d_b = batch['hint_2d_b'] 
        pose_3d = batch['pose_3d'] 
        rotation = batch['rotation']

        # motion encode
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)
        text = batch["text"]
        # classifier free guidance: randomly drop text during training
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        # text encode
        cond_emb = self.text_encoder(text)
        
        n_set = self._diffusion_control_process_2d(z, cond_emb, lengths,hint_2d=hint_2d_a,pose=pose_3d)
        
        hint_mask = hint.sum(-1) != 0
        pose_mask = pose_3d.sum(-1) != 0
        with torch.no_grad():
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask).detach()
        
        controlnet_cond_2d_a = self.traj_encoder_2d(hint_2d_a, mask=hint_mask)
        controlnet_cond_2d_b = self.traj_encoder_2d(hint_2d_b, mask=hint_mask)

        # cond_pose_3d = self.pose_encoder_2d(pose_3d, mask=pose_mask)

        # controlnet_cond_2d_a = controlnet_cond_2d_a #+ cond_pose_3d

        loss_dict = dict()
        loss_dict['info_nce_loss'] = self.info_nce_loss(controlnet_cond,controlnet_cond_2d_a)
        loss_dict["embedding_2d_loss"] = torch.tensor(0., device=cond_emb.device)#self.info_nce_loss(controlnet_cond_2d_a,controlnet_cond_2d_b)
        loss_dict["align_loss"] = 0.5*(self.mse_loss(controlnet_cond,controlnet_cond_2d_a) + self.mse_loss(controlnet_cond,controlnet_cond_2d_b))
        # is_pose_loss = False
        is_drop_hint = False

        loss_dict['traj_loss'] = torch.tensor(0., device=loss_dict['info_nce_loss'].device)
        loss_dict['diff_loss'] = torch.tensor(0., device=loss_dict['info_nce_loss'].device)

        if self.denoiser.time_cond_proj_dim is not None:
            # LCM
            model_pred = n_set['model_pred']
            target = n_set['model_gt']
            # Performance comparison: l2 loss > huber loss
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            model_pred = n_set['noise_pred']
            target = n_set['noise']
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        if self.is_controlnet and is_drop_hint == False:
            z_pred = n_set['model_pred']
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)

            joints_rst_2d = self.project_to_2d(joints_rst,rotation)
            joints_rst_2d = self.datamodule.norm_spatial(joints_rst_2d)  
            joints_rst_2d = joints_rst_2d.view(joints_rst_2d.shape[0], joints_rst_2d.shape[1], self.njoints, 3)[...,:2]

            hint_2d_a = hint_2d_a.view(hint_2d_a.shape[0], hint_2d_a.shape[1], self.njoints, 3)[...,:2]  

            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)     

            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3)
            mask_hint = hint.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                cond_loss = (self.l2_loss(joints_rst, hint).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                cond_loss_2d = (self.l2_loss(joints_rst_2d, hint_2d_a).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                loss_dict['traj_loss'] = self.cond_ratio * cond_loss 
                loss_dict['traj_loss_2d'] = self.cond_ratio * cond_loss_2d 

            else:
                loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)
        else:
            loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)

        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict

    def _diffusion_lcm_control_process_2d(self, latents, encoder_hidden_states, lengths=None, pose_2d=None, pose_2d_mask=None, hint=None,hint_2d=None):
        """
        Modified for multi-step training (num_inference_timesteps > 1).
        """
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


        latents = latents.permute(1, 0, 2)

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample multiple timesteps for multi-step training
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, self.scheduler.num_inference_steps),  # num_inference_timesteps > 1
            device=latents.device,
        )
        timesteps = timesteps.sort(dim=1, descending=True)[0]  # Ensure descending order
        noisy_latents = latents.clone()

        total_loss = 0
        predictions = []

        for i in range(self.scheduler.num_inference_steps):
            # Add noise for this timestep
            noisy_latents = self.noise_scheduler.add_noise(noisy_latents, noise, timesteps[:, i])

            # Compute ControlNet residuals
            controlnet_residuals = None
            if self.is_controlnet and controlnet_cond is not None:
                controlnet_residuals = self.controlnet(
                    sample=noisy_latents,
                    timestep=timesteps[:, i],
                    timestep_cond=timestep_cond,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                )

            # Predict noise residual
            noise_pred = self.denoiser(
                sample=noisy_latents,
                timestep=timesteps[:, i],
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                pose_cond=pose_2d,
                pose_mask=pose_2d_mask,
                controlnet_residuals=controlnet_residuals,
            )

            # Predicted latent
            model_pred = self.predicted_origin(noise_pred, timesteps[:, i], noisy_latents)
            predictions.append(model_pred)

            # Update noisy_latents for the next step
            noisy_latents = model_pred.clone()

        # Collect outputs
        n_set = {
            "noise": noise,
            "model_preds": predictions,
            "model_gt": latents,
        }
        return n_set


    def extract_keypose(self,joints,mask):
        mask = mask.float()
        mask_flattened = mask.squeeze(-1).any(dim=-1).float()  # [B, Len]
        print("Mask flattened shape:", mask_flattened.shape)

        assert mask_flattened.sum(dim=1).max() <= 1, "Each sample should have only one keypose frame."

        keypose_indices = mask_flattened.argmax(dim=1)  # [B]
        keypose_joints = joints[torch.arange(joints.size(0)), keypose_indices]  # [B, Joints, 3]

        return keypose_joints

    def calculate_joint_vectors(self, joints):

        return joints[:, 1:] - joints[:, :-1]

    def train_control_lcm_diffusion_traj_keypose2d_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        hint = batch['hint'] if 'hint' in batch else None 
        hint_2d_a = batch['hint_2d_a'] 
        hint_2d_b = batch['hint_2d_b'] 

        pose_3d = batch['pose_3d']
        pose_2d = batch['pose_2d'] 
        pose_2d_b = batch['pose_2d_b'] 

        pose_mask = batch["pose_mask"]
        rotation = batch['rotation']

        # motion encode
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)
        text = batch["text"]
        # classifier free guidance: randomly drop text during training
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        # text encode
        cond_emb = self.text_encoder(text)

        pose_mask = pose_mask.view(pose_mask.shape[0], pose_mask.shape[1], -1)

        n_set = self._diffusion_lcm_control_process_2d(z, cond_emb, lengths,hint_2d=hint_2d_a,pose_2d=pose_2d_b,pose_2d_mask=pose_mask)
        
        hint_mask = hint.sum(-1) != 0
        pose_mask = pose_3d.sum(-1) != 0
        with torch.no_grad():
            pose_cond_3d = self.denoiser.pose_encoder(pose_3d,mask=pose_mask.bool())
        
        with torch.no_grad():
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask).detach()
        
        pose_cond_2d = self.denoiser.pose_encoder_2d(pose_2d_b,mask=pose_mask.bool())

        controlnet_cond_2d_a = self.traj_encoder_2d(hint_2d_a, mask=hint_mask)
        
        ###
        # controlnet_cond_2d_a = self.traj_encoder_2d(hint_2d_b, mask=hint_mask)
        # cond_pose_3d = self.pose_encoder_2d(pose_3d, mask=pose_mask)
        # controlnet_cond_2d_a = controlnet_cond_2d_a #+ cond_pose_3d

        loss_dict = dict()
        loss_dict['info_nce_loss'] = self.info_nce_loss(pose_cond_3d,pose_cond_2d) + self.info_nce_loss(controlnet_cond,controlnet_cond_2d_a)#self.info_nce_loss(controlnet_cond,controlnet_cond_2d_a)
        loss_dict["embedding_2d_loss"] = torch.tensor(0., device=cond_emb.device)#self.info_nce_loss(pose_cond_2d,pose_cond_2d_b)
        
        loss_dict["align_loss"] = self.mse_loss(pose_cond_3d, pose_cond_2d) + self.mse_loss(controlnet_cond, controlnet_cond_2d_a)#self.mse_loss(controlnet_cond,controlnet_cond_2d_a)
        
        is_pose_loss = True
        is_drop_hint = True

        loss_dict['traj_loss'] = torch.tensor(0., device=cond_emb.device)
        loss_dict['diff_loss'] = torch.tensor(0., device=cond_emb.device)

        if self.denoiser.time_cond_proj_dim is not None:
            # LCM
            # model_pred = n_set['model_pred']
            model_preds = n_set['model_preds']
            target = n_set['model_gt']
            # Diffusion loss (average over timesteps)
            diff_loss = 0
            for model_pred in model_preds:
                diff_loss += F.mse_loss(model_pred, target, reduction="mean")
            diff_loss /= self.scheduler.num_inference_steps

            # Performance comparison: l2 loss > huber loss
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            model_pred = n_set['noise_pred']
            target = n_set['noise']
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        # Keypose loss
        if self.is_controlnet and is_pose_loss:
            z_pred = model_preds[-1]  # Use the final step's prediction for keypose loss
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)

            joints_rst_2d = self.project_to_2d(joints_rst, rotation)
            joints_rst_2d = self.datamodule.norm_spatial(joints_rst_2d)  
            joints_rst_2d = joints_rst_2d.view(joints_rst_2d.shape[0], joints_rst_2d.shape[1], self.njoints, 3)[...,:2]

            pose_2d = pose_2d.view(pose_2d.shape[0], pose_2d.shape[1], self.njoints, 3)[...,:2]  

            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)

            pose_temp = pose_3d.view(pose_3d.shape[0], pose_3d.shape[1], self.njoints, 3)
            mask_pose = pose_temp.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                cond_loss = (self.l2_loss(joints_rst, pose_temp).sum(-1, keepdims=True) * mask_pose).sum() / mask_pose.sum()
                cond_loss_2d = (self.l2_loss(joints_rst_2d, pose_2d).sum(-1, keepdims=True) * mask_pose).sum() / mask_pose.sum()

                # mask_rot = lengths_to_mask(lengths, feats_rst.device).unsqueeze(-1)
                # cond_loss_3d_rot = (self.l2_loss(feats_rst, feats_ref) * mask_rot).mean()

                loss_dict['keypose_loss'] = 2 * self.cond_ratio * cond_loss#2
                loss_dict['keypose_loss_2d'] = 2 * self.cond_ratio * cond_loss_2d#2
                # loss_dict['keypose_loss_3d_rot'] = 0.01 * self.cond_ratio * cond_loss_3d_rot

            else:
                loss_dict['keypose_loss'] = torch.tensor(0., device=diff_loss.device)
        else:
            loss_dict['keypose_loss'] = torch.tensor(0., device=diff_loss.device)

        # Trajectory loss
        if self.is_controlnet and is_drop_hint == False:
            z_pred = model_preds[-1]
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)

            joints_rst_2d = self.project_to_2d(joints_rst,rotation)
            joints_rst_2d = self.datamodule.norm_spatial(joints_rst_2d)  
            joints_rst_2d = joints_rst_2d.view(joints_rst_2d.shape[0], joints_rst_2d.shape[1], self.njoints, 3)[...,:2]

            hint_2d_a = hint_2d_a.view(hint_2d_a.shape[0], hint_2d_a.shape[1], self.njoints, 3)[...,:2]  

            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)     

            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3)
            mask_hint = hint.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                cond_loss = (self.l2_loss(joints_rst, hint).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                cond_loss_2d = (self.l2_loss(joints_rst_2d, hint_2d_a).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                loss_dict['traj_loss'] = self.cond_ratio * cond_loss 
                loss_dict['traj_loss_2d'] = self.cond_ratio * cond_loss_2d 

            else:
                loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)
        else:
            loss_dict['traj_loss'] = torch.tensor(0., device=diff_loss.device)

        # Total loss
        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict


    def project_to_2d(self, motion_3d, rotation_matrix):
       
        B, F, J, _ = motion_3d.shape
        
        motion_3d_flat = motion_3d.view(B, -1, 3)  # [B, F*J, 3]
        
        motion_3d_rotated = torch.bmm(motion_3d_flat, rotation_matrix.transpose(1, 2))  # [B, F*J, 3]
                
        motion_2d = motion_3d_rotated.view(B, F, -1)  # [B, F, J, 2]
        
        return motion_2d

    def train_diffusion_forward(self, batch: dict) -> dict:
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # motion encode
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)

        text = batch["text"]
        # classifier free guidance: randomly drop text during training
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        # text encode
        cond_emb = self.text_encoder(text)

        # diffusion process return with noise and noise_pred
        hint = batch['hint'] if 'hint' in batch else None  # control signals
        n_set = self._diffusion_process(z, cond_emb, hint)

        loss_dict = dict()

        if self.denoiser.time_cond_proj_dim is not None:
            # LCM
            model_pred = n_set['model_pred']
            target = n_set['model_gt']
            # Performance comparison: l2 loss > huber loss
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            model_pred = n_set['noise_pred']
            target = n_set['noise']
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        if self.is_controlnet and self.vaeloss:
            z_pred = n_set['model_pred']
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)
            hint = batch['hint']
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3)
            mask_hint = hint.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                if self.vaeloss_type == 'mean':
                    cond_loss = (self.l2_loss(joints_rst, hint) * mask_hint).mean()
                    loss_dict['cond_loss'] = self.cond_ratio * cond_loss
                elif self.vaeloss_type == 'sum':
                    cond_loss = (self.l2_loss(joints_rst, hint).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                    loss_dict['cond_loss'] = self.cond_ratio * cond_loss
                elif self.vaeloss_type == 'mask':
                    cond_loss = self.masked_l2(joints_rst, hint, mask_hint)
                    loss_dict['cond_loss'] = self.cond_ratio * cond_loss
                else:
                    raise ValueError(f'Unsupported vaeloss_type: {self.vaeloss_type}')
            else:
                loss_dict['cond_loss'] = torch.tensor(0., device=diff_loss.device)

            if self.rot_ratio != 0:
                mask_rot = lengths_to_mask(lengths, feats_rst.device).unsqueeze(-1)
                if self.vaeloss_type == 'mean':
                    rot_loss = (self.l2_loss(feats_rst, feats_ref) * mask_rot).mean()
                    loss_dict['rot_loss'] = self.rot_ratio * rot_loss
                elif self.vaeloss_type == 'sum':
                    rot_loss = (self.l2_loss(feats_rst, feats_ref).sum(-1, keepdims=True) * mask_rot).sum() / mask_rot.sum()
                    rot_loss = rot_loss / self.nfeats
                    loss_dict['rot_loss'] = self.rot_ratio * rot_loss
                elif self.vaeloss_type == 'mask':
                    rot_loss = self.masked_l2(feats_rst, feats_ref, mask_rot)
                    loss_dict['rot_loss'] = self.rot_ratio * rot_loss
                else:
                    raise ValueError(f'Unsupported vaeloss_type: {self.vaeloss_type}')
            else:
                loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        else:
            loss_dict['cond_loss'] = torch.tensor(0., device=diff_loss.device)
            loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict

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
            vae_st = time.time()
            feats_rst = self.vae.decode(z, lengths)
            vae_et = time.time()
            self.vae_decode_times.append(vae_et - vae_st)

        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst
        }

        if 'hint' in batch:
            hint = batch['hint']
            mask_hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint = self.datamodule.denorm_spatial(hint)
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3) * mask_hint
            rs_set['hint'] = hint
            rs_set['mask_hint'] = mask_hint
        else:
            rs_set['hint'] = None

        return rs_set

    def t2m_control_eval(self, batch: dict) -> dict:
        texts = batch["text"]
        feats_ref = batch["motion"]
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        pose = batch["pose_3d"]
        pose_mask = batch["pose_mask"].bool()

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
        # z = self._diffusion_reverse(text_emb, hint)

        z = self._diffusion_control_reverse(text_emb,lengths, pose=pose, pose_mask=pose_mask, hint=hint)
        # z = self._diffusion_control_reverse(text_emb,lengths,hint=hint)

        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        with torch.no_grad():
            vae_st = time.time()
            feats_rst = self.vae.decode(z, lengths)
            vae_et = time.time()
            self.vae_decode_times.append(vae_et - vae_st)

        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "joints_mask": pose_mask
        }

        if 'hint' in batch:
            hint = batch['hint']
            mask_hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint = self.datamodule.denorm_spatial(hint)
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3) * mask_hint
            rs_set['hint'] = hint
            rs_set['mask_hint'] = mask_hint
        else:
            rs_set['hint'] = None

        return rs_set

    def t2m_control_eval_2d(self, batch: dict) -> dict:
        texts = batch["text"]
        feats_ref = batch["motion"]
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        pose_3d = batch['pose_3d'] 
        rotation = batch["rotation"]

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
        hint_2d = batch['hint_2d_a'] if 'hint' in batch else None  # control signals
        # z = self._diffusion_reverse(text_emb, hint)

        z = self._diffusion_control_reverse_2d(text_emb,lengths,hint_2d=hint_2d, rotation=rotation)

        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        with torch.no_grad():
            vae_st = time.time()
            feats_rst = self.vae.decode(z, lengths)
            vae_et = time.time()
            self.vae_decode_times.append(vae_et - vae_st)

        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        if joints_rst.shape[0] != rotation.shape[0]:
            rotation = rotation.repeat(joints_rst.shape[0], 1, 1)

        joints_rst_2d = self.project_to_2d(joints_rst,rotation)
        joints_rst_2d = joints_rst_2d.view(joints_rst_2d.shape[0], joints_rst_2d.shape[1], self.njoints, 3)
        hint_2d = self.datamodule.denorm_spatial(hint_2d)
        hint_2d = hint_2d.view(hint_2d.shape[0], hint_2d.shape[1], self.njoints, 3)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "joints_mask": pose_mask
        }

        if 'hint' in batch:
            hint = batch['hint']
            mask_hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint = self.datamodule.denorm_spatial(hint)
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3) * mask_hint

            
            rs_set['hint'] = hint
            rs_set['mask_hint'] = mask_hint

            rs_set['joints_rst_2d'] = joints_rst_2d
            rs_set['hint_2d'] = hint_2d
        else:
            rs_set['hint'] = None

        return rs_set

    def t2m_control_eval_2d_keypose(self, batch: dict) -> dict:
        texts = batch["text"]
        feats_ref = batch["motion"]
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        pose_3d = batch['pose_3d'] 
        rotation = batch["rotation"]

        pose_2d = batch['pose_2d'] 
        pose_mask = batch["pose_mask"].bool()

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
        hint_2d = batch['hint_2d_a'] if 'hint' in batch else None  # control signals
        # z = self._diffusion_reverse(text_emb, hint)

        z = self._diffusion_control_reverse_2d_keypose(text_emb,lengths,hint_2d=hint, rotation=rotation,pose_2d=pose_2d,pose_2d_mask=pose_mask)

        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        with torch.no_grad():
            vae_st = time.time()
            feats_rst = self.vae.decode(z, lengths)
            vae_et = time.time()
            self.vae_decode_times.append(vae_et - vae_st)

        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        if joints_rst.shape[0] != rotation.shape[0]:
            rotation = rotation.repeat(joints_rst.shape[0], 1, 1)

        joints_rst_2d = self.project_to_2d(joints_rst,rotation)
        joints_rst_2d = joints_rst_2d.view(joints_rst_2d.shape[0], joints_rst_2d.shape[1], self.njoints, 3)

        joints_ref_2d = self.project_to_2d(joints_ref,rotation)
        joints_ref_2d = joints_ref_2d.view(joints_ref_2d.shape[0], joints_ref_2d.shape[1], self.njoints, 3)

        hint_2d = self.datamodule.denorm_spatial(hint_2d)
        hint_2d = hint_2d.view(hint_2d.shape[0], hint_2d.shape[1], self.njoints, 3)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "joints_mask": pose_mask
        }

        if 'hint' in batch:
            hint = batch['hint']
            mask_hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint = self.datamodule.denorm_spatial(hint)
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3) * mask_hint

            rs_set['hint'] = hint
            rs_set['mask_hint'] = mask_hint

            rs_set['joints_rst_2d'] = joints_rst_2d
            rs_set['joints_ref_2d'] = joints_ref_2d
            rs_set['hint_2d'] = hint_2d
        else:
            rs_set['hint'] = None

        return rs_set

    def t2m_control_eval_2d_keypose_traj(self, batch: dict) -> dict:
        texts = batch["text"]
        feats_ref = batch["motion"]
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        pose_3d = batch['pose_3d'] 
        rotation = batch["rotation"]

        pose_2d = batch['pose_2d_b'] 
        pose_mask = batch["pose_mask"].bool()

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
        hint_2d = batch['hint_2d_a'] if 'hint' in batch else None  # control signals
        # z = self._diffusion_reverse(text_emb,hint)

        z = self._diffusion_control_reverse_2d_keypose(text_emb,lengths,hint_2d=hint_2d, rotation=rotation,pose_2d=pose_2d,pose_2d_mask=pose_mask)

        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        with torch.no_grad():
            vae_st = time.time()
            feats_rst = self.vae.decode(z, lengths)
            vae_et = time.time()
            self.vae_decode_times.append(vae_et - vae_st)

        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        if joints_rst.shape[0] != rotation.shape[0]:
            rotation = rotation.repeat(joints_rst.shape[0], 1, 1)

        joints_rst_2d = self.project_to_2d(joints_rst,rotation)
        joints_rst_2d = joints_rst_2d.view(joints_rst_2d.shape[0], joints_rst_2d.shape[1], self.njoints, 3)

        joints_ref_2d = self.project_to_2d(joints_ref,rotation)
        joints_ref_2d = joints_ref_2d.view(joints_ref_2d.shape[0], joints_ref_2d.shape[1], self.njoints, 3)

        hint_2d = self.datamodule.denorm_spatial(hint_2d)
        hint_2d = hint_2d.view(hint_2d.shape[0], hint_2d.shape[1], self.njoints, 3)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "joints_mask": pose_mask
        }

        if 'hint' in batch:
            hint = batch['hint']
            mask_hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint = self.datamodule.denorm_spatial(hint)
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3) * mask_hint

            
            rs_set['hint'] = hint
            rs_set['mask_hint'] = mask_hint

            rs_set['joints_rst_2d'] = joints_rst_2d
            rs_set['joints_ref_2d'] = joints_ref_2d
            rs_set['hint_2d'] = hint_2d
        else:
            rs_set['hint'] = None

        return rs_set


    def allsplit_step(self, split: str, batch: dict) -> Optional[dict]:
        if split in ["test", "val"]:
            # rs_set = self.t2m_eval(batch)
            # rs_set = self.t2m_control_eval(batch)
            # rs_set = self.t2m_control_eval_2d_keypose(batch)
            # rs_set = self.t2m_control_eval_2d_keypose_traj(batch)

            # if self.is_retrieval:
            #     rs_set = self.t2m_control_eval_2d_keypose_traj_retrieval(batch)
            # elif self.is_lifting:
            #     rs_set = self.t2m_control_lifting(batch)
            # else:
            rs_set = self.t2m_control_eval_2d_keypose_traj(batch)


            if self.datamodule.is_mm:
                metric_list = ['MMMetrics']
            else:
                metric_list = self.metric_list

            for metric in metric_list:
                if metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"])
                elif metric == "MMMetrics" and self.datamodule.is_mm:
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0), batch["length"])
                elif metric == 'KeyPosMetrics':
                    # assert rs_set['hint'] is not None
                    getattr(self, metric).update(rs_set["joints_ref"], rs_set['joints_rst'],
                                                 rs_set['joints_mask'], batch['length'])
                elif metric == 'KeyPosMetrics2D':
                    # assert rs_set['hint'] is not None
                    getattr(self, metric).update(rs_set["joints_ref"], rs_set['joints_rst'],
                                                 rs_set["joints_ref_2d"], rs_set['joints_rst_2d'],
                                                 rs_set['joints_mask'], batch['length'])
                elif metric == 'ControlMetrics':
                    assert rs_set['hint'] is not None
                    getattr(self, metric).update(rs_set["joints_rst"], rs_set['hint'],
                                                 rs_set['mask_hint'], batch['length'])
                elif metric == 'ControlMetrics2DProj':
                    assert rs_set['hint'] is not None
                    getattr(self, metric).update(rs_set["joints_rst"], rs_set['hint'],rs_set["joints_rst_2d"], rs_set['hint_2d'],
                                                 rs_set['mask_hint'], batch['length'])

                else:
                    raise TypeError(f"Not support this metric: {metric}.")

        if split in ["train", "val"]:
            # loss_dict = self.train_diffusion_forward(batch)
            # loss_dict = self.train_control_diffusion_forward(batch)
            # loss_dict = self.train_control_lcm_diffusion_forward(batch)
            
            # loss_dict = self.train_control_diffusion_traj2d_forward(batch)

            loss_dict = self.train_control_lcm_diffusion_traj_keypose2d_forward(batch)

            return loss_dict



