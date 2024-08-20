import os
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import torchvision.transforms as T
import imageio

class OBJDataset(Dataset):
    def __init__(self, data_path, camera_path):
        super().__init__()
        self.data_path = data_path
        self.camera_path = camera_path
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )    
        self.view_indices = [ 0,  2,  4, 6, 7, 8, 10, 12, 14, 16, 17, 18, 20, 22, 24, 26, 27, 28, 30, 32, 34, 36, 37, 38]
        self.gen_indices = [ 0,  2,  4,  6,  10, 12, 14, 16, 20, 22, 24, 26, 30, 32, 34, 36]        
        self.all_indices = np.arange(0, 40, 1)
        self.all_indices = np.delete(self.all_indices, self.view_indices)

        pose_json_path = self.camera_path
        with open(pose_json_path, 'r') as f:
            meta = json.load(f)
        
        self.img_ids = list(meta["c2ws"].keys()) 
        self.img_ids = [f"{image_name}_gt" if image_name[-1].isdigit() and image_name[-3] == '_' and image_name[-2].isdigit() else image_name for image_name in self.img_ids]
        self.input_poses = np.array(list(meta["c2ws"].values()))
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.array(meta["intrinsics"])
        self.intrinsics = intrinsic
        self.rgb_paths = [os.path.join(data_path, f"{image_name}.png") for image_name in self.img_ids]
        self.rgb_paths = np.array(self.rgb_paths)
        self.c2ws = []
        self.w2cs = []

        for idx, img_id in enumerate(self.img_ids):
            pose = self.input_poses[idx]
            c2w = pose @ self.blender2opencv
            self.c2ws.append(c2w)
            self.w2cs.append(np.linalg.inv(c2w))

        self.c2ws = np.stack(self.c2ws, axis=0)
        self.w2cs = np.stack(self.w2cs, axis=0)

        self.define_transforms()

        self.z_near = 0.5
        self.z_far = 1.8

    def __len__(self):
        return 1
    
    def define_transforms(self):
        self.img_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
        )
        self.mask_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0,), (1.0,))]
        )

    def __getitem__(self, index):
        src_rgb_paths = self.rgb_paths[self.gen_indices]
        src_w2c_mats = self.w2cs[self.gen_indices]
        src_intrinsics = np.array([self.intrinsics] * len(src_rgb_paths))
        
        src_rgbs = []
        for rgb_path in src_rgb_paths:
            img = imageio.imread(rgb_path)
            mask = self.mask_transforms(img[..., 3])
            rgb = self.img_transforms(img[..., :3])
            rgb = rgb * mask + (1.0 - mask)

            h, w = rgb.shape[-2:]

            src_rgbs.append(rgb)

        train_rgb_paths = self.rgb_paths[self.view_indices]
        train_c2w_mats = self.c2ws[self.view_indices]
        train_w2c_mats = self.w2cs[self.view_indices]
        train_intrinsics = np.array([self.intrinsics] * len(train_rgb_paths))

        train_bbox = []
        train_rgbs = []
        for rgb_path in train_rgb_paths:
            img = imageio.imread(rgb_path)
            mask = self.mask_transforms(img[..., 3])
            rgb = self.img_transforms(img[..., :3])
            rgb = rgb * mask + (1.0 - mask)

            yy = torch.any(mask, axis=2)
            xx = torch.any(mask, axis=1)
            ynz = torch.nonzero(yy)[:, 1]
            xnz = torch.nonzero(xx)[:, 1]
            ymin, ymax = ynz[[0, -1]]
            xmin, xmax = xnz[[0, -1]]
            bbox = torch.FloatTensor([xmin, ymin, xmax, ymax])
            train_bbox.append(bbox)

            train_rgbs.append(rgb)

        all_rgb_paths = self.rgb_paths[self.all_indices]
        all_c2w_mats = self.c2ws[self.all_indices]
        all_intrinsics = np.array([self.intrinsics] * len(all_rgb_paths))

        all_rgbs = []
        for rgb_path in all_rgb_paths:
            img = imageio.imread(rgb_path)
            mask = self.mask_transforms(img[..., 3])
            rgb = self.img_transforms(img[..., :3])
            rgb = rgb * mask + (1.0 - mask)

            all_rgbs.append(rgb)

        depth_range = np.array([self.z_near, self.z_far])

        return {
            "src_rgbs": torch.stack(src_rgbs).permute([0, 2, 3, 1]).float().cuda(), 
            "src_w2c_mats": torch.FloatTensor(src_w2c_mats).cuda(),
            "src_intrinsics": torch.FloatTensor(src_intrinsics).cuda(),
            "train_rgbs": torch.stack(train_rgbs).permute([0, 2, 3, 1]).float().cuda(),
            "train_c2w_mats": torch.FloatTensor(train_c2w_mats).cuda(),
            "train_w2c_mats": torch.FloatTensor(train_w2c_mats).cuda(),
            "train_intrinsics": torch.FloatTensor(train_intrinsics).cuda(),
            "all_rgbs": torch.stack(all_rgbs).permute([0, 2, 3, 1]).float(),
            "all_c2w_mats": torch.FloatTensor(all_c2w_mats).cuda(),
            "all_intrinsics": torch.FloatTensor(all_intrinsics).cuda(),
            "depth_range": torch.FloatTensor(depth_range).cuda(),
            "train_bbox": torch.stack(train_bbox),
        }