import torch as th
import torch.nn as nn
import torch.nn.functional as F



class TriplaneConv(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, is_rollout=True) -> None:
        super().__init__()
        in_channels = channels * 3 if is_rollout else channels
        self.is_rollout = is_rollout

        self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        if self.is_rollout:
            tpl_xy_h = th.cat([tpl_xy,
                            th.mean(tpl_yz, dim=-1, keepdim=True).transpose(-1, -2).expand_as(tpl_xy),
                            th.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_xy)], dim=1) # [B, C * 3, H, W]
            tpl_xz_h = th.cat([tpl_xz,
                                th.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_xz),
                                th.mean(tpl_yz, dim=-2, keepdim=True).expand_as(tpl_xz)], dim=1) # [B, C * 3, H, D]
            tpl_yz_h = th.cat([tpl_yz,
                            th.mean(tpl_xy, dim=-2, keepdim=True).transpose(-1, -2).expand_as(tpl_yz),
                            th.mean(tpl_xz, dim=-2, keepdim=True).expand_as(tpl_yz)], dim=1) # [B, C * 3, W, D]
        else:
            tpl_xy_h = tpl_xy
            tpl_xz_h = tpl_xz
            tpl_yz_h = tpl_yz
        
        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        tpl_xy_h = self.conv_xy(tpl_xy_h)
        tpl_xz_h = self.conv_xz(tpl_xz_h)
        tpl_yz_h = self.conv_yz(tpl_yz_h)

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)                
    

class TriplaneNorm(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.norm_xy = nn.GroupNorm(32, channels)
        self.norm_xz = nn.GroupNorm(32, channels)
        self.norm_yz = nn.GroupNorm(32, channels)

    def forward(self, featmaps):
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy_h = self.norm_xy(tpl_xy) # [B, C, H, W]
        tpl_xz_h = self.norm_xz(tpl_xz) # [B, C, H, D]
        tpl_yz_h = self.norm_yz(tpl_yz) # [B, C, W, D]

        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)        


class TriplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        return (self.silu(tpl_xy), self.silu(tpl_xz), self.silu(tpl_yz))
    

class TriplaneUpsample2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy = F.interpolate(tpl_xy, scale_factor=2, mode='bilinear', align_corners=False)
        tpl_xz = F.interpolate(tpl_xz, scale_factor=2, mode='bilinear', align_corners=False)
        tpl_yz = F.interpolate(tpl_yz, scale_factor=2, mode='bilinear', align_corners=False)

        assert tpl_xy.shape[-2] == H * 2 and tpl_xy.shape[-1] == W * 2
        assert tpl_xz.shape[-2] == H * 2 and tpl_xz.shape[-1] == D * 2
        assert tpl_yz.shape[-2] == W * 2 and tpl_yz.shape[-1] == D * 2

        return (tpl_xy, tpl_xz, tpl_yz)
    

class TriplaneResBlock(nn.Module):

    def __init__(
        self,
        channels,
        out_channels,
        use_conv=False,
        is_rollout=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            TriplaneNorm(channels),
            TriplaneSiLU(),
            TriplaneConv(channels, self.out_channels, 3, padding=1, is_rollout=is_rollout),
        )


        self.h_upd = TriplaneUpsample2x()
        self.x_upd = TriplaneUpsample2x()

        self.out_layers = nn.Sequential(
            TriplaneNorm(self.out_channels),
            TriplaneSiLU(),
            TriplaneConv(self.out_channels, self.out_channels, 3, padding=1, is_rollout=is_rollout),
            TriplaneConv(self.out_channels, self.out_channels, 3, padding=1, is_rollout=is_rollout)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = TriplaneConv(
                channels, self.out_channels, 3, padding=1, is_rollout=False
            )
        else:
            self.skip_connection = TriplaneConv(channels, self.out_channels, 1, padding=0, is_rollout=False)

    def forward(self, x):

        h = self.in_layers(x)
        h = self.out_layers(h)

        x_skip = self.skip_connection(x)
        x_skip_xy, x_skip_xz, x_skip_yz = x_skip
        h_xy, h_xz, h_yz = h
        return (h_xy + x_skip_xy, h_xz + x_skip_xz, h_yz + x_skip_yz)


class TriplaneSR(nn.Module):

    def __init__(
        self,
        in_channels=128,
        out_channels=[64, 64, 32],
        use_fp16=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = th.float16 if use_fp16 else th.float32

        self.in_conv = TriplaneConv(in_channels, in_channels, 1, padding=0, is_rollout=False)
        self.in_res = TriplaneResBlock(in_channels, in_channels)
        
        # -L version
        self.in_res_1 = TriplaneResBlock(in_channels, in_channels)
        self.in_res_2 = TriplaneResBlock(in_channels, in_channels)

        self.up1 = TriplaneUpsample2x()
        self.res1 = TriplaneResBlock(in_channels, out_channels[0])
        self.res1_1 = TriplaneResBlock(out_channels[0], out_channels[0])
        self.res1_2 = TriplaneResBlock(out_channels[0], out_channels[0])

        self.out0 = nn.Sequential(
            TriplaneNorm(in_channels),
            TriplaneSiLU(),
            TriplaneConv(in_channels, out_channels[2], 1, padding=0, is_rollout=False),
            TriplaneNorm(out_channels[2]),
            TriplaneSiLU(),
            TriplaneConv(out_channels[2], out_channels[2], 1, padding=0, is_rollout=False),
            TriplaneNorm(out_channels[2]),
            TriplaneSiLU(),
            TriplaneConv(out_channels[2], out_channels[2], 1, padding=0, is_rollout=False)                        
        )

        self.out1 = nn.Sequential(
            TriplaneNorm(out_channels[0]),
            TriplaneSiLU(),
            TriplaneConv(out_channels[0], out_channels[2], 1, padding=0, is_rollout=False),
            TriplaneNorm(out_channels[2]),
            TriplaneSiLU(),
            TriplaneConv(out_channels[2], out_channels[2], 1, padding=0, is_rollout=False),
            TriplaneNorm(out_channels[2]),
            TriplaneSiLU(),
            TriplaneConv(out_channels[2], out_channels[2], 1, padding=0, is_rollout=False)                        
        )

    def forward(self, x):
        h_triplane0 = self.in_conv(x)
        h_triplane0 = self.in_res(h_triplane0)
        h_triplane0 = self.in_res_1(h_triplane0)
        h_triplane0out = self.in_res_2(h_triplane0)

        h_triplane1 = self.up1(h_triplane0)
        h_triplane1 = self.res1(h_triplane1)
        h_triplane1 = self.res1_1(h_triplane1)
        h_triplane1out = self.res1_2(h_triplane1)

        h_triplane0put = self.out0(h_triplane0out)
        h_triplane1put = self.out1(h_triplane1out)
        
        return [h_triplane0put, h_triplane1put]