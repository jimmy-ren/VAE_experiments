import torch.nn as nn
import torch.nn.functional as F
from sympy.physics.units import temperature

from aux_funs import *

class NormActConv(nn.Module):
    """
    Perform GroupNorm, Activation, and Convolution operations.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int = 8,
                 kernel_size: int = 3,
                 norm: bool = True,
                 act: bool = True,
                 dropout: bool = True,
                 dropout_prob: float = 0.1
                 ):
        super(NormActConv, self).__init__()

        # GroupNorm
        self.g_norm = nn.GroupNorm(
            num_groups,
            in_channels
        ) if norm is True else nn.Identity()

        # Activation
        self.act = nn.SiLU() if act is True else nn.Identity()

        self.dropout = nn.Dropout(p=dropout_prob) if dropout else nn.Identity()

        # Convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

class SelfAttentionBlock(nn.Module):
    """
    Perform GroupNorm and Multiheaded Self Attention operation.
    """

    def __init__(self,
                 num_channels: int,
                 num_groups: int = 8,
                 num_heads: int = 4,
                 norm: bool = True
                 ):
        super(SelfAttentionBlock, self).__init__()

        # GroupNorm
        self.g_norm = nn.GroupNorm(
            num_groups,
            num_channels
        ) if norm is True else nn.Identity()

        # Self-Attention
        self.attn = nn.MultiheadAttention(
            num_channels,
            num_heads,
            batch_first=True
        )

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h * w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x

class Downsample(nn.Module):
    """
    Perform Downsampling by the factor of k across Height and Width.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int = 2,  # Downsampling factor
                 use_conv: bool = True,  # If Downsampling using conv-block
                 use_mpool: bool = True  # If Downsampling using max-pool
                 ):
        super(Downsample, self).__init__()

        self.use_conv = use_conv
        self.use_mpool = use_mpool

        # Downsampling using Convolution
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(
                in_channels,
                out_channels // 2 if use_mpool else out_channels,
                kernel_size=4,
                stride=k,
                padding=1
            )
        ) if use_conv else nn.Identity()

        # Downsampling using Maxpool
        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k),
            nn.Conv2d(
                in_channels,
                out_channels // 2 if use_conv else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if use_mpool else nn.Identity()

    def forward(self, x):

        if not self.use_conv:
            return self.mpool(x)

        if not self.use_mpool:
            return self.cv(x)

        return torch.cat([self.cv(x), self.mpool(x)], dim=1)


class Upsample(nn.Module):
    """
    Perform Upsampling by the factor of k across Height and Width
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int = 2,  # Upsampling factor
                 use_conv: bool = True,  # Upsampling using conv-block
                 use_upsample: bool = False  # Upsampling using nn.upsample
                 ):
        super(Upsample, self).__init__()

        self.use_conv = use_conv
        self.use_upsample = use_upsample

        # Upsampling using conv
        self.cv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels // 2 if use_upsample else out_channels,
                kernel_size=4,
                stride=k,
                padding=1
            ),
            nn.Conv2d(
                out_channels // 2 if use_upsample else out_channels,
                out_channels // 2 if use_upsample else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if use_conv else nn.Identity()

        # Upsamling using nn.Upsample
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=k,
                mode='bilinear',
                align_corners=False
            ),
            nn.Conv2d(
                in_channels,
                out_channels // 2 if use_conv else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if use_upsample else nn.Identity()

    def forward(self, x):

        if not self.use_conv:
            return self.up(x)

        if not self.use_upsample:
            return self.cv(x)

        return torch.cat([self.cv(x), self.up(x)], dim=1)


class DownC(nn.Module):
    """
    Perform Down-convolution on the input using following approach.
    1. Conv + TimeEmbedding
    2. Conv
    3. Skip-connection from input x.
    4. Self-Attention
    5. Skip-Connection from 3.
    6. Downsampling
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,  # Time Embedding Dimension
                 num_layers: int = 2,
                 down_sample: bool = True,  # True for Downsampling
                 enable_attention: bool = True
                 ):
        super(DownC, self).__init__()

        self.num_layers = num_layers
        self.enable_attention = enable_attention

        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels,
                        out_channels
                        ) for i in range(num_layers)
        ])

        self.conv2 = nn.ModuleList([
            NormActConv(out_channels,
                        out_channels
                        ) for _ in range(num_layers)
        ])

        self.te_block = nn.ModuleList([
            #TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])

        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])

        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()

        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=1
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        out = x

        for i in range(self.num_layers):
            resnet_input = out

            # Resnet Block
            out = self.conv1[i](out)
            #out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            if self.enable_attention:
                # Self Attention
                out_attn = self.attn_block[i](out)
                out = out + out_attn

        # Downsampling
        out = self.down_block(out)

        return out

class MidAndSampling(nn.Module):

    def __init__(self, latent_dim):
        super(MidAndSampling, self).__init__()
        self.fc1 = nn.Linear(256*7*7, 256)
        self.fc2 = nn.Linear(256, latent_dim*2)
        self.fc3 = nn.Linear(latent_dim, latent_dim*2)
        self.fc4 = nn.Linear(latent_dim*2, 256)
        self.fc5 = nn.Linear(256, 256*7*7)

    def forward(self, x):
        # x is in the form of ncwh
        x = x.view(-1, 256*7*7)
        x = F.relu(self.fc1(x))
        # predicts mean µ and log(σ²)
        pred_params = self.fc2(x)
        # mean µ
        pred_mean = pred_params[:, 0:int(pred_params.size(1) / 2)]
        # log(σ²)
        pred_log_variance = pred_params[:, int(pred_params.size(1) / 2):]
        # σ²
        pred_variance = torch.exp(pred_log_variance)
        # σ
        pred_sd = torch.sqrt(pred_variance)
        # draw sample from standard Gaussian
        latent_samples = torch.randn(pred_params.size(0), int(pred_params.size(1) / 2))
        latent_samples = latent_samples.to(pred_params.device)
        # re-parametrization
        latent_samples = pred_mean + latent_samples * pred_sd
        x = F.relu(self.fc3(latent_samples))
        x = F.relu(self.fc4(x))
        out = F.relu(self.fc5(x))
        out = out.view(-1, 256, 7, 7)

        return pred_mean, pred_variance, out

class KernelPrediction(nn.Module):

    def __init__(self, latent_dim):
        super(KernelPrediction, self).__init__()

    def forward(self, latent_embedding, input_img, filter_bank):
        coord = latent_embedding[0]
        normalized_coord = normalize_mu(coord, assumed_range=(-3, 3))
        grid_coords = normalized_coord.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # duplicate the filter bank for batch_size times
        filter_bank = filter_bank.repeat(input_img.size()[0], 1, 1, 1, 1)

        # Perform grid sampling
        sampled_filters = F.grid_sample(
            input=filter_bank,  # [100, 49, 8, 3, 3] - the filter bank
            grid=grid_coords,  # [100, 1, 1, 1, 3] - normalized coordinates
            align_corners=True,  # Important: maps exactly to grid corners
            mode='bilinear',  # Actually trilinear for 3D
            padding_mode='border'  # How to handle out-of-bound coordinates
        )

        final_filters = sampled_filters.squeeze(-1).squeeze(-1).squeeze(-1)

        input_img = input_img.view(-1, 49)
        net_out = torch.linalg.vecdot(input_img, final_filters, dim=1)

        return final_filters, net_out
        #return net_out

class MidC(nn.Module):
    """
    Refine the features obtained from the DownC block.
    It refines the features using following operations:

    1. Resnet Block with Time Embedding
    2. A Series of Self-Attention + Resnet Block with Time-Embedding
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,
                 num_layers: int = 2,
                 enable_attention: bool = True
                 ):
        super(MidC, self).__init__()

        self.num_layers = num_layers
        self.enable_attention = enable_attention

        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels,
                        out_channels
                        ) for i in range(num_layers + 1)
        ])

        self.conv2 = nn.ModuleList([
            NormActConv(out_channels,
                        out_channels
                        ) for _ in range(num_layers + 1)
        ])

        self.te_block = nn.ModuleList([
            #TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1)
        ])

        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])

        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=1
            ) for i in range(num_layers + 1)
        ])

    def forward(self, x, t_emb):
        out = x

        # First-Resnet Block
        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out = out + self.res_block[0](resnet_input)

        # Sequence of Self-Attention + Resnet Blocks
        for i in range(self.num_layers):
            if self.enable_attention:
                # Self Attention
                out_attn = self.attn_block[i](out)
                out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.conv1[i + 1](out)
            out = out + self.te_block[i + 1](t_emb)[:, :, None, None]
            out = self.conv2[i + 1](out)
            out = out + self.res_block[i + 1](resnet_input)

        return out


class UpC(nn.Module):
    """
    Perform Up-convolution on the input using following approach.
    1. Upsampling
    2. Conv + TimeEmbedding
    3. Conv
    4. Skip-connection from 1.
    5. Self-Attention
    6. Skip-Connection from 3.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,  # Time Embedding Dimension
                 num_layers: int = 2,
                 up_sample: bool = True,  # True for Upsampling
                 enable_attention: bool = True
                 ):
        super(UpC, self).__init__()

        self.num_layers = num_layers
        self.enable_attention = enable_attention

        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels,
                        out_channels
                        ) for i in range(num_layers)
        ])

        self.conv2 = nn.ModuleList([
            NormActConv(out_channels,
                        out_channels
                        ) for _ in range(num_layers)
        ])

        self.te_block = nn.ModuleList([
            #TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])

        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])

        #self.up_block = Upsample(in_channels, in_channels // 2) if up_sample else nn.Identity()
        self.up_block = Upsample(in_channels, in_channels) if up_sample else nn.Identity()

        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=1
            ) for i in range(num_layers)
        ])

    def forward(self, x, down_out):
        # Upsampling
        x = self.up_block(x)
        #x = torch.cat([x, down_out], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out

            # Resnet Block
            out = self.conv1[i](out)
            #out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)\

            if self.enable_attention:
                # Self Attention
                out_attn = self.attn_block[i](out)
                out = out + out_attn

        return out

class KLD_VAE_Loss(nn.Module):
    def __init__(self):
        super(KLD_VAE_Loss, self).__init__()

    def forward(self, pred_mean, pred_variance):
        loss = -torch.log(torch.sqrt(pred_variance)) + (pred_variance + pred_mean.pow(2)) / 2 - 0.5
        #tmp = -0.5 * torch.sum(1 + torch.log(pred_variance) - pred_mean.pow(2) - pred_variance)
        loss = torch.sum(loss)
        return loss