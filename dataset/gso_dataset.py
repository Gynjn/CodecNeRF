import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import imageio
import cv2
import json
import tqdm
from torch.utils.data import Dataset

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

def intrinsic_to_fov(K, w=None, h=None):
    # Extract the focal lengths from the intrinsic matrix
    fx = K[0, 0]
    fy = K[1, 1]
    
    w = K[0, 2]*2 if w is None else w
    h = K[1, 2]*2 if h is None else h
    
    # Calculating field of view
    fov_x = 2 * np.arctan2(w, 2 * fx) 
    fov_y = 2 * np.arctan2(h, 2 * fy)
    
    return fov_x, fov_y

class GSODataset(Dataset):
    def __init__(self, data_path, camera_path):
        super().__init__()
        self.data_path = data_path
        scenes_name = np.array([f for f in sorted(os.listdir(self.data_path)) if os.path.isdir(f'{self.data_path}/{f}')])[300:]
        self.scenes_name =  scenes_name
        self.img_size = np.array([256, 256])

        self.camera_path = camera_path
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ) 
        self.z_near = 0.5
        self.z_far = 1.8
        # self.z_far = 2.5
        self.build_metas()        

    def build_metas(self):

        self.scene_infos = {}

        for scene in tqdm.tqdm(self.scenes_name):
                
            json_info = json.load(open(os.path.join(self.data_path, scene,f'transforms.json')))
            b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            scene_info = {'ixts': [], 'c2ws': [], 'w2cs':[], 'img_paths': [], 'fovx': [], 'fovy':[]}
            positions = []
            for idx, frame in enumerate(json_info['frames']):
                c2w = np.array(frame['transform_matrix'])
                c2w = c2w @ b2c
                ixt = np.array(frame['intrinsic_matrix'])
                fov_x, fov_y = intrinsic_to_fov(ixt)
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['c2ws'].append(c2w.astype(np.float32))
                scene_info['w2cs'].append(np.linalg.inv(c2w.astype(np.float32)))
                img_path = os.path.join(self.data_path, scene, f'r_{idx:03d}.png')
                scene_info['img_paths'].append(img_path)
                scene_info['fovx'].append(fov_x)
                scene_info['fovy'].append(fov_y)
                positions.append(c2w[:3,3])
            
            self.scene_infos[scene] = scene_info

    def __len__(self):
        return len(self.scene_infos)
    
    def define_transforms(self):
        self.img_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
        )
        self.mask_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0,), (1.0,))]
        )

    def read_view(self, scene, idx):
        img, mask = self.read_image(scene, idx)
        ixt, ext = self.read_cam(scene, idx)
        return img, mask, ext, ixt
    
    def read_views(self, scene, src_views, bg_color):
        src_ids = src_views
        ixts, exts, w2cs, imgs, msks= [], [], [], [], []
        for idx in src_ids:
            img, mask = self.read_image(scene, idx, bg_color)
            imgs.append(img)
            ixt, ext, w2c = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            msks.append(mask)
        return np.stack(imgs), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts)

    def read_cam(self, scene, view_idx):
        img_downscale = np.array([256, 256])/512
        ext = scene['c2ws'][view_idx]
        w2c = scene['w2cs'][view_idx]
        ixt = scene['ixts'][view_idx].copy()
        ixt[:2] =  ixt[:2] * img_downscale.reshape(2,1)
        return ixt, ext, w2c

    def read_image(self, scene, view_idx, bg_color):
        img_path = scene['img_paths'][view_idx]
        img = imageio.imread(img_path)
        img = cv2.resize(img, (256, 256))
        mask = (img[...,-1] > 0).astype('uint8')
        img = img.astype(np.float32) / 255.
        img = (img[..., :3] * img[..., -1:] + bg_color*(1 - img[..., -1:])).astype(np.float32)
        return img, mask


    def __getitem__(self, index):

        scene_name = self.scenes_name[index]
        scene_info = self.scene_infos[scene_name]
        bg_color = np.ones(3).astype(np.float32)
        tar_views = list(range(32))
        all_indices = np.arange(0, 32, 1)
        view_indices = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31]
        gen_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        all_indices = np.delete(all_indices, view_indices)

        tar_img, tar_msks, tar_c2ws, tar_w2cs, tar_ixts = self.read_views(scene_info, tar_views, bg_color)
        tar_c2ws[:, :3, 3] = tar_c2ws[:, :3, 3]*(2/3)
        tar_w2cs[:, :3, 3] = tar_w2cs[:, :3, 3]*(2/3)

        near_far = np.array([0.5, 1.8]).astype(np.float32)
        return {
            "scene_name": scene_name,
            "src_rgbs": torch.from_numpy(tar_img[gen_list]).cuda(),
            "src_w2c_mats": torch.from_numpy(tar_w2cs[gen_list]).cuda(),
            "src_intrinsics": torch.from_numpy(tar_ixts[gen_list]).cuda(),
            "train_rgbs": torch.from_numpy(tar_img[view_indices]).cuda(),
            "train_c2w_mats": torch.from_numpy(tar_c2ws[view_indices]).cuda(),
            "train_w2c_mats": torch.from_numpy(tar_w2cs[view_indices]).cuda(),
            "train_intrinsics": torch.from_numpy(tar_ixts[view_indices]).cuda(),
            "all_rgbs": torch.from_numpy(tar_img[all_indices]),
            "all_c2w_mats": torch.from_numpy(tar_c2ws[all_indices]).cuda(),
            "all_intrinsics": torch.from_numpy(tar_ixts[all_indices]).cuda(),
            "depth_range": torch.from_numpy(near_far).cuda(),
        }