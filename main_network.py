from aux_modules import *
from aux_funs import *

class Unet(nn.Module):
    """
    U-net architecture which is used to predict noise
    in the paper "Denoising Diffusion Probabilistic Model".

    U-net consists of Series of DownC blocks followed by MidC
    followed by UpC.
    """

    def __init__(self,
                 im_channels: int = 1, # RGB
                 down_ch: list = [32, 64, 128, 256],
                 mid_ch: list = [256, 256, 256],
                 up_ch: list[int] = [256, 128, 64, 16],
                 down_sample: list[bool] = [False, False, False],
                 t_emb_dim: int = 128,
                 num_downc_layers :int = 2,
                 num_midc_layers :int = 2,
                 num_upc_layers :int = 2,
                 enable_attention:bool = True
                 ):
        super(Unet, self).__init__()

        self.im_channels = im_channels
        self.down_ch = down_ch
        self.mid_ch = mid_ch
        self.up_ch = up_ch
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.num_downc_layers = num_downc_layers
        self.num_midc_layers = num_midc_layers
        self.num_upc_layers = num_upc_layers
        self.enable_attention = enable_attention

        self.up_sample = list(reversed(self.down_sample)) # [False, True, True]

        # Initial Convolution
        self.cv1 = nn.Conv2d(self.im_channels, self.down_ch[0], kernel_size=3, padding=1)

        # Initial Time Embedding Projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # DownC Blocks
        self.downs = nn.ModuleList([
            DownC(
                self.down_ch[i],
                self.down_ch[i+1],
                self.t_emb_dim,
                self.num_downc_layers,
                self.down_sample[i],
                self.enable_attention
            ) for i in range(len(self.down_ch) - 1)
        ])

        # MidC Block
        '''
        self.mids = nn.ModuleList([
            MidC(
                self.mid_ch[i],
                self.mid_ch[i+1],
                self.t_emb_dim,
                self.num_midc_layers,
                self.enable_attention
            ) for i in range(len(self.mid_ch) - 1)
        ])
        '''

        latent_dim = 3
        self.mid_layers = MidAndSampling(latent_dim)
        self.kpn_layers = KernelPrediction(latent_dim)

        # UpC Block
        self.ups = nn.ModuleList([
            UpC(
                self.up_ch[i],
                self.up_ch[i+1],
                self.t_emb_dim,
                self.num_upc_layers,
                self.up_sample[i],
                self.enable_attention
            ) for i in range(len(self.up_ch) - 1)
        ])

        # Final Convolution
        self.cv2 = nn.Sequential(
            nn.GroupNorm(8, self.up_ch[-1]),
            nn.Conv2d(self.up_ch[-1], self.im_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, filter_bank, buddy_filter=None):

        out = self.cv1(x)

        # DownC outputs
        down_outs = []

        for down in self.downs:
            down_outs.append(out)
            out = down(out)

        # MidC outputs
        #for mid in self.mids:
        #    out = mid(out, t_emb)
        pred_mean, pred_variance, out = self.mid_layers(out)

        # KPN Block
        ker_pred, filter_out = self.kpn_layers([pred_mean, pred_variance], x, filter_bank, buddy_filter)

        # UpC Blocks
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out)

        # Final Conv
        out = self.cv2(out)

        return pred_mean, pred_variance, out, ker_pred, filter_out