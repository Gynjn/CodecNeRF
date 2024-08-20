import types
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.checkpoint import checkpoint
from network.tri_block import TriplaneGen
from network.spa_block import Spatial3DNet
from network.tri_new import TriplaneSR
from network.tri_res import TriplaneGroupResnetBlock
from vector_quantize_pytorch import VectorQuantize
from network.attention import attentionblock
from dahuffman import HuffmanCodec
import numpy as np

from network.encoders.dino_wrapper import DinoWrapper
from einops import rearrange

def construct_project_matrix(x_ratio, y_ratio, K, w2cs):
    rfn = K.shape[0] # Batchsize
    K = K[:, :3, :3]
    scale_m = torch.tensor([x_ratio, y_ratio, 1.0], dtype=torch.float32, device=K.device)
    scale_m = torch.diag(scale_m) # 3, 3
    ref_prj = scale_m[None, :, :] @ K @ w2cs[:, :3] # rfn, 3, 4
    pad_vals = torch.zeros([rfn, 1, 4], dtype=torch.float32, device=K.device)
    pad_vals[:, :, 3] = 1.0
    ref_prj = torch.cat([ref_prj, pad_vals], 1)  # rfn,4,4
    return ref_prj


def project_and_normalize(ref_grid, src_proj, length_y, length_x):
    src_grid = src_proj[:, :3, :3] @ ref_grid + src_proj[:, :3, 3:] # b 3 n
    div_val = src_grid[:, -1:]
    div_val[div_val<1e-4] = 1e-4
    src_grid = src_grid[:, :2] / div_val
    src_grid[:, 0] = src_grid[:, 0]/((length_x - 1) / 2) - 1 # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1]/((length_y - 1) / 2) - 1 # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1) # (b, n, 2)
    return src_grid    


def get_warp_coordinates(volume_xyz, warp_h, warp_w, input_h, input_w, K, warp_pose):
    B, _, D, H, W = volume_xyz.shape
    ratio_x = warp_w / input_w
    ratio_y = warp_h / input_h
    warp_proj = construct_project_matrix(ratio_x, ratio_y, K, warp_pose) # B, 4, 4
    warp_coords = project_and_normalize(volume_xyz.view(B, 3, D*H*W), warp_proj, warp_h, warp_w).view(B, D, H, W, 2)
    return warp_coords


class Codec(nn.Module):
    def __init__(
            self,
            input_dim_2d,
            extractor_channel,
            input_dim_3d,
            dims_3d,
            input_dim_tri,
            feat_dim_tri,
            spatial_volume_length,
            V,
            device,
    ):
        
        super(Codec, self).__init__()
        self.input_dim_2d = input_dim_2d
        self.extractor_channel = extractor_channel
        self.input_dim_3d = input_dim_3d
        self.dims_3d = dims_3d
        self.input_dim_tri = input_dim_tri
        self.feat_dim_tri = feat_dim_tri
        self.spatial_volume_length = spatial_volume_length
        self.V = V
        self.device=device

        self.dinov2 = DinoWrapper(model_name='facebook/dino-vitb16', freeze=False,)
        self.detockenize_layer = nn.ConvTranspose2d(768, 16, kernel_size=16, stride=16)

        self.spatial_conv = Spatial3DNet(input_dim=self.input_dim_3d, dims=dims_3d)
        self.tri_gen = TriplaneGen(n_channels=self.input_dim_tri, feat_dim=32)
        self.aware_net = TriplaneSR(in_channels=feat_dim_tri, out_channels=[64, 64, 64])

        self.group_res_down1 = TriplaneGroupResnetBlock(in_channels=32, out_channels=32)        
        self.group_res_down2 = TriplaneGroupResnetBlock(in_channels=32, out_channels=32)

        self.group_res_up1 = TriplaneGroupResnetBlock(in_channels=32, out_channels=64) 
        self.group_res_up2 = TriplaneGroupResnetBlock(in_channels=64, out_channels=64)      

        self.vq = VectorQuantize(dim=32, codebook_size=8192, decay=0.8, commitment_weight=0,)
        self.vq_bit = math.log2(8192)

        self.at0 = attentionblock(in_ch=64)
        self.at1 = attentionblock(in_ch=64)
        self.atb0 = attentionblock(in_ch=32)
        self.atb1 = attentionblock(in_ch=32)

        self.gr0 = TriplaneGroupResnetBlock(in_channels=64, out_channels=128)
        self.gr01 = TriplaneGroupResnetBlock(in_channels=128, out_channels=128)
        self.gr02 = TriplaneGroupResnetBlock(in_channels=128, out_channels=128)
        self.gr03 = TriplaneGroupResnetBlock(in_channels=128, out_channels=256)
        self.gr04 = TriplaneGroupResnetBlock(in_channels=256, out_channels=256)
        self.gr05 = TriplaneGroupResnetBlock(in_channels=256, out_channels=256)
        self.gr06 = TriplaneGroupResnetBlock(in_channels=256, out_channels=128)
        self.gr07 = TriplaneGroupResnetBlock(in_channels=128, out_channels=128)
        self.gr08 = TriplaneGroupResnetBlock(in_channels=128, out_channels=64)
        self.gr09 = TriplaneGroupResnetBlock(in_channels=64, out_channels=64)
        self.gr10 = TriplaneGroupResnetBlock(in_channels=64, out_channels=32)
        self.g0 = TriplaneGroupResnetBlock(in_channels=32, out_channels=32)
        self.g00 = TriplaneGroupResnetBlock(in_channels=32, out_channels=32)

        self.gr1 = TriplaneGroupResnetBlock(in_channels=64, out_channels=128)
        self.gr11 = TriplaneGroupResnetBlock(in_channels=128, out_channels=128)
        self.gr12 = TriplaneGroupResnetBlock(in_channels=128, out_channels=128)
        self.gr13 = TriplaneGroupResnetBlock(in_channels=128, out_channels=256)
        self.gr14 = TriplaneGroupResnetBlock(in_channels=256, out_channels=256)
        self.gr15 = TriplaneGroupResnetBlock(in_channels=256, out_channels=256)
        self.gr16 = TriplaneGroupResnetBlock(in_channels=256, out_channels=128)
        self.gr17 = TriplaneGroupResnetBlock(in_channels=128, out_channels=128)
        self.gr18 = TriplaneGroupResnetBlock(in_channels=128, out_channels=64)
        self.gr19 = TriplaneGroupResnetBlock(in_channels=64, out_channels=64)
        self.gr20 = TriplaneGroupResnetBlock(in_channels=64, out_channels=32)
        self.g1 = TriplaneGroupResnetBlock(in_channels=32, out_channels=32)
        self.g11 = TriplaneGroupResnetBlock(in_channels=32, out_channels=32)


    def huffman_encode(self, param):
        input_code_list = param.view(-1).tolist()
        unique, counts= np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        total_mb = total_bits/8/10**6
        return total_mb


    def forward(self, x, K, poses, w2cs):
        """
        x: (16, 3, 128, 128)
        K: (16, 4, 4)
        poses: (16, 4, 4)
        """
        B, N, H, W, C = x.shape
        V = self.V
        input_image_h = H
        input_image_w = W
        x = rearrange(x, 'B N C H W -> (B N) C H W')
        extracted_out = self.dinov2(x) #(B, h*w/(16**2), 768)
        extracted_out = extracted_out[:,1:]
        extracted_out = extracted_out.permute(0,2,1).reshape(-1, 768, 16, 16)
        extracted_out = self.detockenize_layer(extracted_out) #(B*N, 16, 256, 256)


        _, NC, NH, NW = extracted_out.shape
        extracted_out = extracted_out.reshape([B, N, NC, NH, NW]) 

        spatial_volume_verts = torch.linspace(-self.spatial_volume_length, self.spatial_volume_length, V, dtype=torch.float32, device=self.device)
        spatial_volume_verts = torch.stack(torch.meshgrid(spatial_volume_verts, spatial_volume_verts, spatial_volume_verts), -1)
        spatial_volume_verts = spatial_volume_verts.reshape(1, V ** 3, 3)[:, :, (2, 1, 0)]
        spatial_volume_verts = spatial_volume_verts.view(1, V, V, V, 3).permute(0, 4, 1, 2, 3).repeat(B, 1, 1, 1, 1)

        spatial_volume_feats = []
        for ni in range(0, N):
            w2c_source = w2cs[:, ni]
            ex_source = extracted_out[:, ni] 
            K_source = K[:, ni]
            C = ex_source.shape[1]
            coord_source = get_warp_coordinates(spatial_volume_verts, ex_source.shape[-2], ex_source.shape[-1],
                                                input_image_h, input_image_w, K_source, w2c_source).view(B, V, V*V, 2)
            unproj_feats = F.grid_sample(ex_source, coord_source, mode='bilinear', padding_mode='zeros', align_corners=True)
            unproj_feats = unproj_feats.view(B, C, V, V, V)
            spatial_volume_feats.append(unproj_feats)

        spatial_volume_feats = torch.stack(spatial_volume_feats, 1)
        N = spatial_volume_feats.shape[1]
        spatial_volume_feats = spatial_volume_feats.view(B, N*C, V, V, V)
        spatial_volume_feats = self.spatial_conv(spatial_volume_feats)
        xy_feat, xz_feat, yz_feat = self.tri_gen(spatial_volume_feats)

        feat_map_ = self.atb0(self.group_res_down1((xy_feat, xz_feat, yz_feat)))
        feat_map_ = self.atb1(self.group_res_down2(feat_map_))

        xy_feat = feat_map_[0]
        xz_feat = feat_map_[1]
        yz_feat = feat_map_[2]
        feat_map_ = torch.stack([xy_feat, xz_feat, yz_feat], dim=1) # (B, N, F, R1, R2)

        Batch, N_plane, Feat_dim, R1, R2 = feat_map_.shape
        feat_map_ = feat_map_.permute(0, 1, 3, 4, 2)
        feat_map_ = feat_map_.reshape(Batch, -1, feat_map_.shape[-1])

        quantized, indices, commit_loss = self.vq(feat_map_)

        code_mb = self.huffman_encode(indices)

        quantized = quantized.view(Batch, N_plane, R1, R2, Feat_dim)

        xy_feat = quantized[:, 0].permute(0, 3, 1, 2)
        xz_feat = quantized[:, 1].permute(0, 3, 1, 2)
        yz_feat = quantized[:, 2].permute(0, 3, 1, 2)

        feat_map_ = self.at0(self.group_res_up1((xy_feat, xz_feat, yz_feat)))
        feat_map_ = self.at1(self.group_res_up2(feat_map_))

        feat_map0, feat_map1 = self.aware_net((feat_map_[0], feat_map_[1], feat_map_[2]))

        feat_map0 = self.gr04(self.gr03(self.gr02(self.gr01(self.gr0(feat_map0)))))
        feat_map0 = self.gr10(self.gr09(self.gr08(self.gr07(self.gr06(self.gr05(feat_map0))))))

        feat_map1 = self.gr14(self.gr13(self.gr12(self.gr11(self.gr1(feat_map1)))))
        feat_map1 = self.gr20(self.gr19(self.gr18(self.gr17(self.gr16(self.gr15(feat_map1))))))
        
        feat_map0 = self.g00(self.g0(feat_map0))
        feat_map1 = self.g11(self.g1(feat_map1))

        xy_feat0 = feat_map0[0]
        xz_feat0 = feat_map0[1]
        yz_feat0 = feat_map0[2]

        xy_feat1 = feat_map1[0]
        xz_feat1 = feat_map1[1]
        yz_feat1 = feat_map1[2]

        feat_map0 = torch.stack([xy_feat0, yz_feat0, xz_feat0], dim=0)
        feat_map1 = torch.stack([xy_feat1, yz_feat1, xz_feat1], dim=0)

        return [feat_map0, feat_map1], commit_loss, code_mb, indices