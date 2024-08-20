import torch.nn as nn
import torch.nn.functional as F

class TriplaneConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channel: int, # but if you are doing 1x1 conv it should be an integer
            ker_size=3,
            stride=1,
            padd=0,
            use_norm=None
    ):
        
        super(TriplaneConv, self).__init__()
        in_c = in_channels
        out_c = out_channel
        self.conv_yz = nn.Conv2d(in_c, out_c, ker_size, stride, padd) # Originally out_c
        self.conv_xz = nn.Conv2d(in_c, out_c, ker_size, stride, padd)
        self.conv_xy = nn.Conv2d(in_c, out_c, ker_size, stride, padd)

    def forward(self, tri_feats: list, tri_noises: list=None, add_noise=False, skip_add=False):
        """
        Args:
        tri_feats (list): tri-plane feature maps
        tri_noises (list, optional): tri-plane noise maps
        add_noise (bool, optional): add_tri-lane noise maps
        skip_add (bool, optional): skip connection

        Returns:
        output: tri-plane feature maps
        """
        yz_feat, xz_feat, xy_feat = tri_feats

        if skip_add:
            yz_feat_prev = yz_feat
            xz_feat_prev = xz_feat
            xy_feat_prev = xy_feat

        if add_noise and tri_noises is not None:
            yz_feat = yz_feat + tri_noises[0]
            xz_feat = xz_feat + tri_noises[1]
            xy_feat = xy_feat + tri_noises[2]
        
        yz_feat = self.conv_yz(yz_feat)
        xz_feat = self.conv_xz(xz_feat)
        xy_feat = self.conv_xy(xy_feat)

        # skip connect
        if skip_add:
            yz_feat = yz_feat + yz_feat_prev
            xz_feat = xz_feat + xz_feat_prev
            xy_feat = xy_feat + xy_feat_prev

        return yz_feat, xz_feat, xy_feat
    

class TriplaneGen(nn.Module):
    def __init__(self, n_channels, feat_dim):

        super(TriplaneGen, self).__init__()
        self.n_channels = n_channels
        self.feat_dim = feat_dim
        self.pool_dim = 1


        self.head_conv = TriplaneConv(self.n_channels, self.feat_dim, 1, 1, 0)

    def forward(self, spatial_volume):
        ni = spatial_volume
        in_shape = ni.shape[-3:] # 32, 32, 32
        yz_feat = F.adaptive_avg_pool3d(ni, (self.pool_dim, in_shape[1], in_shape[2])).squeeze(2)
        xz_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], self.pool_dim, in_shape[2])).squeeze(3)
        xy_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], in_shape[1], self.pool_dim)).squeeze(4)

        yz_feat, xz_feat, xy_feat = self.head_conv([yz_feat, xz_feat, xy_feat])

        return xy_feat, xz_feat, yz_feat