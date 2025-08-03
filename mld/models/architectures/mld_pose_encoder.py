import torch
import torch.nn as nn
from torch import  nn
from mld.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from mld.models.operator import PositionalEncoding

from mld.models.operator.cross_attention import (SkipTransformerEncoder,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional, Union
from torch.distributions.distribution import Distribution
from torch.optim import AdamW
from torch.nn import Parameter
import math
from inspect import isfunction

# from GraphMotion.models.architectures import transformer
class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)

class PositionNet(nn.Module):
    def __init__(self,  in_dim=768, out_dim=1024, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*2 # 2 is sin&cos, 4 is xyxy 

        self.linears = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, timeline, masks, positive_embeddings):
        timeline = timeline.squeeze(-1)
        B, N, _ = timeline.shape 
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(timeline) # B*N*4 --> B*N*C

        # learnable null embedding 
        positive_null = self.null_positive_feature.view(1,1,-1)
        xyxy_null =  self.null_position_feature.view(1,1,-1)

        # replace padding with learnable null embedding 
        # print("positive_embeddings",positive_embeddings.shape)
        # positive_embeddings = positive_embeddings*masks + (1-masks)*positive_null
        # xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null
        xyxy_embedding = xyxy_embedding.permute(1,0,2)
        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([N,B,self.out_dim])        
        return objs

class ContextTransformer(nn.Module):
    def __init__(self):
        super(ContextTransformer, self).__init__()
        # self.config = config

        # self.d_mask = config["d_mask"]
        # self.constrained_slices = [
        #     slice(*i) for i in config["constrained_slices"]
        # ]

        self.dropout = 0.0 #config["dropout"]
        self.pre_lnorm = True #config["pre_lnorm"]
        self.n_layer = 6 #config["n_layer"]

        self.d_encoder_in = 256
        self.d_encoder_h = 256#512
        self.d_model = 256#512
        self.d_out = 256#135
        self.d_head = 64
        self.n_head = 8
        self.atten_bias = False
        self.d_pff_inner = 512

        self.encoder = nn.Sequential(
            nn.Linear(self.d_encoder_in, self.d_encoder_h),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_encoder_h, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.config["d_model"], self.config["d_decoder_h"]),
        #     nn.PReLU(),
        #     nn.Linear(self.config["d_decoder_h"], self.config["d_out"])
        # )

        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, self.d_head),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_head, self.d_head),
            nn.Dropout(self.dropout)
        )

        self.keyframe_pos_layer = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(self.dropout)
        )

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.att_layers = nn.ModuleList()
        # self.cross_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.RelMultiHeadedAttention(
                    self.n_head, self.d_model,
                    self.d_head, dropout=self.dropout,
                    pre_lnorm=self.pre_lnorm,
                    bias=self.atten_bias
                )
            )

            # self.cross_layers.append(
            #     transformer.MultiHeadedAttention(
            #         self.n_head, self.d_model,
            #         self.d_head, dropout=self.dropout,
            #         pre_lnorm=self.pre_lnorm,
            #         bias=self.atten_bias
            #     )
            # )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.d_model, self.d_pff_inner,
                    dropout=self.dropout,
                    pre_lnorm=self.pre_lnorm
                )
            )

    def get_rel_pos_emb(self, window_len, dtype, device):
        pos_idx = torch.arange(-window_len + 1, window_len,
                               dtype=dtype, device=device)
        pos_idx = pos_idx[None, :, None]        # (1, seq, 1)
        rel_pos_emb = self.rel_pos_layer(pos_idx)
        return rel_pos_emb

    def forward(self, x, text_embedding=None, keyframe_pos=None, mask=None):
        x = self.encoder(x)

        # x = x + self.keyframe_pos_layer(keyframe_pos)

        rel_pos_emb = self.get_rel_pos_emb(x.shape[-2], x.dtype, x.device)

        for i in range(self.n_layer):
            x = self.att_layers[i](x, rel_pos_emb, mask=mask)
            # x = self.cross_layers[i](x,text_emb=text_embedding, mask=mask)

            x = self.pff_layers[i](x)
        if self.pre_lnorm:
            x = self.layer_norm(x)

        # x = self.decoder(x)

        return x



def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))


class MotionEncoder(nn.Module):
    def __init__(self,
                 latent_dim: list = [1, 1024],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:
        
        super().__init__()
        self.latent_dim = latent_dim[-1]
        self.skel_embedding = nn.Linear(264, self.latent_dim)
        self.latent_size = latent_dim[0]
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))
        # self.text_encoded_dim = text_encoded_dim
        # self.condition = condition
        self.abl_plus = False

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        
        self.pe_type = "mld" #ablation.DIFF_PE_TYPE

        encoder_layer_s = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
        encoder_norm = nn.LayerNorm(self.latent_dim)

        self.encoder = TransformerEncoder(encoder_layer_s, num_layers,encoder_norm)

        self.dropout = 0.1
        # self.rel_pos_layer = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.PReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(64, 64),
        #     nn.Dropout(self.dropout)
        # )

    def forward(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            mask=None,
            skip = False,
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]
        
        device = features.device
        # features = torch.cat((features,mask),2)

        features = features.squeeze(2)
        bs, nframes, nfeats = features.shape
        
        
        x = features.float()

        # Embed each human poses into latent vectors
        if skip == False:
            x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        mask = lengths_to_mask(lengths, device)
        # mask = torch.squeeze(mask, dim=-1)

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)

                
        aug_mask = torch.cat((dist_masks, mask), 1)
        # aug_mask = mask

        # adding the embedding token for all sequences

        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos(xseq)
        # rel_pos_emb = self.get_rel_pos_emb(xseq.shape[-2], xseq.dtype, xseq.device)

        # dist = self.encoder(xseq,text_embedding,rel_pos_emb,src_key_padding_mask=~aug_mask)
        # atten_mask = self.get_attention_mask(aug_mask)

        # dist = self.context_transformer(xseq.permute(1, 0, 2),text_embedding).permute(1, 0, 2)
        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask)

        feat = dist[0:2]

        return feat

    

    def configure_optimizers(self):
        optimizer = AdamW(params=filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
        return optimizer


class PoseEncoder(nn.Module):
    def __init__(self,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:
        
        super().__init__()
        self.latent_dim = latent_dim[-1]
        self.skel_embedding = nn.Linear(66, self.latent_dim)
        self.latent_size = latent_dim[0]
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 1, self.latent_dim))
        # self.text_encoded_dim = text_encoded_dim
        # self.condition = condition
        self.abl_plus = False

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        
        self.pe_type = "mld" #ablation.DIFF_PE_TYPE

        encoder_layer_s = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
        encoder_norm = nn.LayerNorm(self.latent_dim)

        self.encoder = TransformerEncoder(encoder_layer_s, num_layers,encoder_norm)

        self.dropout = 0.1
        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, 64),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 64),
            nn.Dropout(self.dropout)
        )

    def forward(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            mask=None,
            skip = False,
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]
        
        device = features.device
        # features = torch.cat((features,mask),2)

        features = features.squeeze(2)
        bs, nframes, nfeats = features.shape
        
        
        x = features.float()

        # Embed each human poses into latent vectors
        if skip == False:
            x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim

        # mask = lengths_to_mask(lengths, device)
        mask = torch.squeeze(mask, dim=-1)

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)

        
        aug_mask = torch.cat((dist_masks, mask), 1)
        # aug_mask = mask

        # adding the embedding token for all sequences

        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos(xseq)
        # rel_pos_emb = self.get_rel_pos_emb(xseq.shape[-2], xseq.dtype, xseq.device)

        # dist = self.encoder(xseq,text_embedding,rel_pos_emb,src_key_padding_mask=~aug_mask)
        # atten_mask = self.get_attention_mask(aug_mask)

        # dist = self.context_transformer(xseq.permute(1, 0, 2),text_embedding).permute(1, 0, 2)
        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask)

        feat = dist[0:1]

        return feat

    

    def configure_optimizers(self):
        optimizer = AdamW(params=filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
        return optimizer


class TimeTableEmbedder(nn.Module):
    def __init__(self,
                 input_dim: int = 196,
                 out_dim: int = 256,
                 position_embedding: str = "learned",
                #  **kwargs
                 ) -> None:
        
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.skel_embedding = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, self.out_dim),
        )


        # self.linear = self.skel_embedding = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.SiLU(),
        #     nn.Linear( 512, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, 768),
        # )

        self.query_pos = build_position_encoding(
            self.input_dim, position_embedding=position_embedding)
       
    def forward(
            self,
            features: Tensor,
            action_features: Tensor,
            lengths: Optional[List[int]] = None,
            mask=None,
            skip = False,
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]
        
        device = features.device
        features = features.squeeze(3)
        bs, nframes, nfeats = features.shape
        
        x = features.float()
        x = x.permute(1, 0, 2)

        feat = self.skel_embedding(x)

     
        feat = torch.cat([action_features,feat],dim=-1)
        return feat


class TimeTableEncoder(nn.Module):
    def __init__(self,
                 latent_dim: list = [1, 768],
                 ff_size: int = 1024,
                 num_layers: int = 2,
                 num_heads: int = 2,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:
        
        super().__init__()
        self.latent_dim = latent_dim[-1]
        self.skel_embedding = nn.Linear(196, self.latent_dim)
        self.latent_size = latent_dim[0]
        # self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 1, self.latent_dim))
        # self.text_encoded_dim = text_encoded_dim
        # self.condition = condition
        self.abl_plus = False

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding="relative")
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        
        self.pe_type = "mld" #ablation.DIFF_PE_TYPE

        encoder_layer_s = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
        encoder_norm = nn.LayerNorm(self.latent_dim)

        self.encoder = TransformerEncoder(encoder_layer_s, num_layers,encoder_norm)

        self.dropout = 0.1

        self.linears = nn.Sequential(
            nn.Linear( 768, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, 768),
        )

        self.cross_attention = GatedCrossAttentionDense(query_dim=768,context_dim=768,n_heads=4,d_head=self.latent_dim)

    def forward(
            self,
            features: Tensor,
            action_features: Tensor,
            lengths: Optional[List[int]] = None,
            mask=None,
            skip = False,
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]
        
        device = features.device
        features = features.squeeze(3)
        bs, nframes, nfeats = features.shape
        
        x = features.float()
        # Embed each human poses into latent vectors
        if skip == False:
            x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        mask = lengths_to_mask(lengths, device)

        # Each batch has its own set of tokens
                
        # aug_mask = torch.cat((dist_masks, mask), 1)
        aug_mask = mask

        # adding the embedding token for all sequences

        xseq = x
        xseq = self.query_pos(xseq)

        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask)
        feat = dist[0:4]

        objs = self.linears(self.cross_attention(feat,action_features))
        # objs = self.linears(torch.cat([action_features, feat], dim=-1))

        return objs


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context):

        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(context)  # B*M*(H*C)
        v = self.to_v(context)  # B*M*(H*C)

        B, N, HC = q.shape
        _, M, _ = k.shape  # M是context的长度
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C
        v = v.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*M
        attn = sim.softmax(dim=-1)  # (B*H)*N*M

        out = torch.einsum('b i j, b j c -> b i c', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)


class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

      

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class GatedSelfAttentionDense2(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        # self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs):
        x = x.permute(1,0,2)
        objs = objs.permute(1,0,2)

        x = x.repeat(1,2,1)
        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape

        # objs = self.linear(objs)
      
        # sanity check 
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # select grounding token and resize it to visual token size as residual 
        out = self.attn(  self.norm1(torch.cat([x,objs],dim=1))  )[:,N_visual:,:]
        out = out.permute(0,2,1).reshape( B,-1,size_g,size_g )
        out = torch.nn.functional.interpolate(out, (size_v,size_v), mode='bicubic')
        residual = out.reshape(B,-1,N_visual).permute(0,2,1)
        
        # add residual to visual feature 
        x = x + self.scale*torch.tanh(self.alpha_attn) * residual
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        x = x[:,:2,:]
        return x.permute(1,0,2)

class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        # we need a linear projection since we need cat visual feature and obj feature
        # self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, objs):

        x = x.permute(1,0,2)

        objs = objs.permute(1,0,2)

        N_visual = x.shape[1]
        # objs = self.linear(objs)
        attention_output = self.attn( self.norm1(torch.cat([x,objs],dim=1)))
        x = x + self.scale*torch.tanh(self.alpha_attn) * attention_output[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        return x.permute(1,0,2) 

class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        self.attn = CrossAttention(query_dim=query_dim, context_dim=context_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        self.scale = 1

    def forward(self, x, objs):
        x = x.permute(1, 0, 2)  # (N, B, D) -> (B, N, D)
        objs = objs.permute(1, 0, 2)  # (M, B, D) -> (B, M, D)

        attention_output = self.attn(self.norm1(x), self.norm1(objs))
        x = x + self.scale * torch.tanh(self.alpha_attn) * attention_output
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x.permute(1, 0, 2)  # (B, N, D) -> (N, B, D)

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'