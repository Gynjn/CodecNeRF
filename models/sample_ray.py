import numpy as np
import torch
import torch.nn.functional as F
from random import randint


rng = np.random.RandomState(234)


def bbox_sample(bboxes, N_rand):
    """
    Args:
        bboxes: bounding box value (xmin, ymin, xmax, ymax) [batch, 4]
        N_rand: number of pixels to sample
    Returns:
        Pixel indices to sample from
    """
    x = (
        torch.rand(N_rand) * (bboxes[2] + 1 - bboxes[0])
        + bboxes[0]
    ).long()
    y = (
        torch.rand(N_rand) * (bboxes[3] + 1 - bboxes[1])
        + bboxes[1]
    ).long()
    return y, x

def bbox_sample_full(bboxes, N_rand, h=128, w=128, prob=0.8):
    """Bounding box sampling but includes other parts of the images
    Args:
        bboxes: bounding box value (xmin, ymin, xmax, ymax) [batch, 4]
        N_rand: number of pixels to sample
        h: image height
        w: image width
        prob: probability of choosing samples inside the bbox
    Returns:
        Pixel indices to sample from
    """
    N_in = int(N_rand * prob)
    N_out = N_rand - N_in

    x = (
        torch.rand(N_in) * (bboxes[2] + 1 - bboxes[0])
        + bboxes[0]
    ).long()
    y = (
        torch.rand(N_in) * (bboxes[3] + 1 - bboxes[1])
        + bboxes[1]
    ).long()


    x_out = (
        torch.rand(N_out) * w
    ).long()

    y_out = (
        torch.rand(N_out) * h
    ).long()

    y = torch.cat([y, y_out])
    x = torch.cat([x, x_out])

    return y, x

########################################################################################################################
# ray batch sampling
########################################################################################################################

class RaySamplerSingleImage(object):
    def __init__(self, data, device, image_res, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data['tgt_rgb'] if 'tgt_rgb' in data.keys() else None # (1, 128, 128, 3)
        self.intrinsics = data['tgt_intrinsic'] # (1, 4, 4)
        self.c2w_mat = data['tgt_c2w_mat'] # (1, 4, 4)
        # self.rgb_path = data['rgb_path']
        self.depth_range = data['depth_range'] # tensor([[0.8, 1.8]])
        self.device = device
        self.batch_size = len(self.intrinsics)

        self.H = image_res[0]
        self.W = image_res[1]

        # for the target
        self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)

        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3) # for the target (16384, 3)

        if 'src_rgbs' in data.keys():
            self.src_rgbs = data['src_rgbs'] # (1, 1, 128, 128, 3)
        else:
            self.src_rgbs = None
        if 'src_masks' in data.keys():
            self.src_masks = data['src_masks'] # (1, 1, 128, 128, 1)
        else:
            self.src_masks = None
        if 'src_intrinsics' in data.keys():
            self.src_intrinsics = data['src_intrinsics'] # (1, 1, 4, 4)
        else:
            self.src_intrinsics = None
        if 'src_c2w_mats' in data.keys():
            self.src_c2w_mats = data['src_c2w_mats'] # (1, 1, 4, 4)
        else:
            self.src_c2w_mats = None    
        if 'tgt_bbox' in data.keys():
            self.tgt_bbox = data['tgt_bbox'] # (1, 4)

        if 'src_rgbs_multi' in data.keys():
            self.src_rgbs_multi = data['src_rgbs_multi'] # (1, 16, 128, 128, 3)
        else:
            self.src_rgbs_multi = None
        if 'src_masks_multi' in data.keys():
            self.src_masks_multi = data['src_masks_multi'] # (1, 16, 128, 128, 1)
        else:
            self.src_masks_multi = None
        if 'src_intrinsics_multi' in data.keys():
            self.src_intrinsics_multi = data['src_intrinsics_multi'] # (1, 16, 4, 4)
        else:
            self.src_intrinsics_multi = None
        if 'src_c2w_mats_multi' in data.keys():
            self.src_c2w_mats_multi = data['src_c2w_mats_multi'] # (1, 16, 4, 4)
        if 'src_w2c_mats_multi' in data.keys():
            self.src_w2c_mats_multi = data['src_w2c_mats_multi'] # (1, 16, 4, 4)

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        '''Generate rays for a single image (batch size = 1).
        
        Args:
            H: image height
            W: image width
            intrinsics: 4 by 4 intrinsic matrix
            c2w: 4 by 4 camera to world extrinsic matrix
        Returns:
            Tensors of ray origin and direction.
        '''
        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32) + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)

        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        rays_d = torch.nn.functional.normalize(rays_d, dim=2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
        return rays_o, rays_d

    def get_all(self):
        ret = {'ray_o': self.rays_o.cuda(),
               'ray_d': self.rays_d.cuda(),
               'depth_range': self.depth_range.cuda(),
               'intrinsics': self.intrinsics.cuda(),
               'c2w_mat': self.c2w_mat.cuda(),
               'rgb': self.rgb.cuda() if self.rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_intrinsics': self.src_intrinsics.cuda() if self.src_intrinsics is not None else None,
               'src_c2w_mats': self.src_c2w_mats.cuda() if self.src_c2w_mats is not None else None,
               'src_masks': self.src_masks.cuda() if self.src_masks is not None else None,
               'src_rgbs_multi': self.src_rgbs_multi.cuda() if self.src_rgbs_multi is not None else None,
               'src_intrinsics_multi': self.src_intrinsics_multi.cuda() if self.src_intrinsics_multi is not None else None,
               'src_masks_multi': self.src_masks_multi.cuda() if self.src_masks_multi is not None else None,            
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                               np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'uniform':
            # Random from one image
            select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_sample(self):
        '''Generate a bundle of randomly sampled rays.

        Args:
            N_rand: number of rays to be casted
        Returns:
            A dictionary of ray information.
        '''
        rays_o = self.rays_o
        rays_d = self.rays_d
        if self.rgb is not None:
            rgb = self.rgb
        else:
            rgb = None

        ret = {'ray_o': rays_o.cuda(),
               'ray_d': rays_d.cuda(),
               'intrinsics': self.intrinsics.cuda(),
               'c2w_mat': self.c2w_mat.cuda(),
               'depth_range': self.depth_range.cuda(),
               'rgb': rgb.cuda() if rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_intrinsics': self.src_intrinsics.cuda() if self.src_intrinsics is not None else None,
               'src_c2w_mats': self.src_c2w_mats.cuda() if self.src_c2w_mats is not None else None,
            #    'selected_inds': select_inds,
               'src_masks': self.src_masks.cuda() if self.src_masks is not None else None,
        }
        return ret

class RaySamplerMultipleImages(object):
    """Ray sampler for multiple images (batch size > 1)
    """
    def __init__(self, data, device, image_res, resize_factor=1, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data['train_rgbs'] if 'train_rgbs' in data.keys() else None # [b, h, w, 3]
        self.intrinsics = data['train_intrinsics'] # [b, 4, 4]
        self.c2w_mat = data['train_c2w_mats'] # [b, 4, 4]
        self.depth_range = data['depth_range'] # [b, 2]
        self.batch_size = len(self.intrinsics)
        self.device = device

        self.H = image_res[0]
        self.W = image_res[1]

        self.c2w_mat = self.c2w_mat.squeeze()
        self.intrinsics = self.intrinsics.squeeze()

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)

        self.rays_o, self.rays_d = self.get_rays_multiple_images(self.H, self.W, self.intrinsics, self.c2w_mat) 
        self.rays_o = self.rays_o.reshape(-1, 3)
        self.rays_d = self.rays_d.reshape(-1, 3)        
        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)

        if 'src_rgbs' in data.keys():
            self.src_rgbs = data['src_rgbs'] # (B, 16, 128, 128, 3)
        else:
            self.src_rgbs = None
        if 'src_masks' in data.keys():
            self.src_masks = data['src_masks'] # (B, 16, 128, 128, 1)
        else:
            self.src_masks = None
        if 'src_intrinsics' in data.keys():
            self.src_intrinsics = data['src_intrinsics'] # (B, 16, 4, 4)
        else:
            self.src_intrinsics = None
        if 'src_c2w_mats' in data.keys():
            self.src_c2w_mats = data['src_c2w_mats'] # (B, 16, 4, 4)
        else:
            self.src_c2w_mats = None
        if 'src_w2c_mats' in data.keys():
            self.src_w2c_mats = data['src_w2c_mats'] # (B, 16, 4, 4)
        else:
            self.src_w2c_mats = None
        if 'train_bbox' in data.keys():
            self.tgt_bbox = data['train_bbox'].squeeze(0) # (B, 4)
        

    def get_rays_multiple_images(self, H, W, intrinsics, c2w):
        # Get the world space rays_o and rays_d
        '''Generate rays for multiple images (batch size > 1).
        Args:
            H: image height
            W: image width
            intrinsics: 4 by 4 intrinsic matrix
            c2w: 4 by 4 camera to world extrinsic matrix
        Returns:
            Tensors of ray origin and direction.
        '''
        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32) + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(24, 1, 1).to(self.device)
        # batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2) # B x HW x 3
        rays_d = torch.nn.functional.normalize(rays_d, dim=2)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[1], 1)  # B x HW x 3

        return rays_o, rays_d

    def get_all(self):
        ret = {'ray_o': self.rays_o.cuda(), # [b, h*w, 3]
               'ray_d': self.rays_d.cuda(), # [b, h*w, 3]
               'depth_range': self.depth_range.cuda(), # [b, 2]
               'intrinsics': self.intrinsics.cuda(), # [b, 4, 4]
               'c2w_mat': self.c2w_mat.cuda(), # [b, 4, 4]
               'rgb': self.rgb.cuda() if self.rgb is not None else None, # [b, h*w, 3]
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None, # [b, v, h, w, 3]
               'src_intrinsics': self.src_intrinsics.cuda() if self.src_intrinsics is not None else None, # [b, v, 4, 4]
               'src_c2w_mats': self.src_c2w_mats.cuda() if self.src_c2w_mats is not None else None, # [b, v, 4, 4]
               'src_w2c_mats': self.src_w2c_mats.cuda() if self.src_w2c_mats is not None else None, # [b, v, 4, 4]
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, batch_idx, center_ratio=0.8):
        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                               np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'bbox':
            u, v = bbox_sample(self.tgt_bbox[batch_idx], N_rand)
            select_inds = v + self.W * u
        elif sample_mode == 'bbox_sample_full':
            u, v = bbox_sample_full(self.tgt_bbox[batch_idx], N_rand, h=self.H, w=self.W, prob=0.8) # u = y, v = x
            select_inds = v + self.W * u
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    # def random_sample(self, N_rand):
    def random_sample(self, N_rand):
        '''Generate a bundle of randomly sampled rays.

        Args:
            N_rand: number of rays to be casted
        Returns:
            A dictionary of ray information.
        '''
        indices = torch.randint(self.rays_d.shape[0], size = [N_rand])
        rays_o, rays_d = self.rays_o[indices], self.rays_d[indices]
        rgb = self.rgb[indices]

        ret = {'ray_o': rays_o.cuda(), # (1, 512, 3)
               'ray_d': rays_d.cuda(), # (1, 512, 3)
               'intrinsics': self.intrinsics.cuda(), # (1, 4, 4)
               'c2w_mat': self.c2w_mat.cuda(), # (1, 4, 4)
               'depth_range': self.depth_range.cuda(), # (1, 2)
               'rgb': rgb.cuda() if rgb is not None else None, # (1, 512, 3)
        }
        return ret
