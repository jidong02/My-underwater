import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from guided_diffusion.script_util import create_model as create_prior_model

logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))

####################
# define denoiser
####################

class ResidualNoiseFusion(nn.Module):
    def __init__(self, in_ch=3, hidden_ch=64, t_dim=128):
        super().__init__()

        # prior projector P
        self.prior_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, in_ch, 3, 1, 1)
        )

        # task projector
        self.task_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, in_ch, 3, 1, 1)
        )

        # gate
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_ch * 3, hidden_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, in_ch, 3, 1, 1),
            nn.Sigmoid()
        )

        # timestep embedding projection
        self.t_proj = nn.Sequential(
            nn.Linear(1, hidden_ch),
            nn.SiLU(),
            nn.Linear(hidden_ch, in_ch)
        )

    def forward(self, eps_task, eps_prior, x_t, t, beta_t):
        # project prior to task-like space
        eps_prior_proj = self.prior_proj(eps_prior)
        eps_task_proj = self.task_proj(eps_task)

        # normalize t to [0,1]
        t_norm = t.float().view(-1, 1) / (t.max().float().clamp(min=1.0))
        t_bias = self.t_proj(t_norm).view(t.shape[0], -1, 1, 1)

        gate_in = torch.cat([eps_task_proj, eps_prior_proj, x_t], dim=1)
        gate = self.gate_net(gate_in)
        gate = torch.clamp(gate + t_bias, 0.0, 1.0)

        eps_mix = eps_task + beta_t * gate * (eps_prior_proj - eps_task)
        return eps_mix

####################
# define network
####################


# Generator
def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'trans':
        from .ddpm_trans_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'trans_div':
        from .ddpm_modules import diffusion, unet_backup
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    if model_opt['which_model_G'] == 'trans_div':
        model = unet_backup.DiT(depth=12, in_channels=6, hidden_size=384, patch_size=4, num_heads=6, input_size=128)
    else:
        model = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )

    fusion_module = ResidualNoiseFusion(
        in_ch=model_opt['diffusion']['channels'],
        hidden_ch=64
    )
   
    prior_model = create_prior_model(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="32,16,8",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
    )

    prior_ckpt = torch.load(opt['model']['prior_state'], map_location='cpu')
    prior_model.load_state_dict(prior_ckpt, strict=True)

    if opt['gpu_ids']:
        prior_model = prior_model.cuda()
    else:
        prior_model = prior_model.cpu()

    prior_model.eval()
    for p in prior_model.parameters():
        p.requires_grad = False

    def prior_eps_fn(x, t):

    # task: 0~1999
    # prior: 0~999

        t_prior = (
            t.float() * 999.0 / float(netG.num_timesteps - 1)
        ).round().long().clamp(0, 999)

        out = prior_model(x, t_prior)

        # guided diffusion learn_sigma=True -> 6 channels
        if out.shape[1] == x.shape[1] * 2:
            out = out[:, :x.shape[1]]

        return out


    # if opt['phase'] == 'train':
    #     # init_weights(netG, init_type='kaiming', scale=0.1)
    #     init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)

    netG.prior_denoise_fn = prior_eps_fn
    netG.fusion_module = fusion_module
    netG.use_fusion = True
    netG.fusion_cfg = dict(
        alpha_early=0.95,
        alpha_mid=0.98,
        alpha_late=1.0,
        early_ratio=0.4,
        mid_ratio=0.4
    )

    return netG

    