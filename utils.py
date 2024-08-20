# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import subprocess
import torch
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2
import os
from datetime import datetime
import shutil
import math
from models.render_image import render_single_image
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import re
import random

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)


def get_single(data_dict):
    new_dict = {}
    for key in data_dict:
        if key == 'img_hw':
            new_dict[key] = [data_dict[key][0][:1], data_dict[key][1][:1]]
        else:
            datum = data_dict[key][0]
            if isinstance(datum, torch.Tensor):
                new_dict[key] = datum.unsqueeze(0)
            else:
                new_dict[key] = [datum]
    return new_dict

def get_views(data_dict, src_indices, tgt_indices):
    """Acquire certain source/target views from a given sample
    
    Args:
        data_dict: sample from eval dataset
        src_indices: source view indices [#views]
        tgt_indices: target view indices [#views]
    
    Returns:
        An array of data_dict's
    """
    samples = []

    """
    For val data
    val_tgt_views = [0, 50, 120]
    val_src_views = [64, 64, 64]
    len(src_indices) = 3 [64, 64, 64]
    """

    """
    For sampling random 16 views
    """

    for i in range(len(src_indices)):
        nearest_ids = np.arange(data_dict['rgbs'].shape[1])
        nearest_ids = np.delete(nearest_ids, tgt_indices[i])
        # Hard parsing 16
        nearest_ids = np.random.choice(nearest_ids, 16, replace=False)
        sample = {
            'rgb_path': data_dict['rgb_path'],
            'img_id': data_dict['img_id'],
            'img_hw': data_dict['img_hw'],
            'depth_range': data_dict['depth_range'],
            'tgt_bbox': data_dict['bbox'][:, tgt_indices[i]],
            'tgt_mask': data_dict['masks'][:, tgt_indices[i]],
            'tgt_rgb': data_dict['rgbs'][:, tgt_indices[i]],
            'tgt_c2w_mat': data_dict['c2w_mats'][:, tgt_indices[i]],
            'tgt_intrinsic': data_dict['intrinsics'][:, tgt_indices[i]],
            'src_masks': data_dict['masks'][:, src_indices[i]][None, :], # HACK assuming one view
            'src_rgbs': data_dict['rgbs'][:, src_indices[i]][None, :],
            'src_c2w_mats': data_dict['c2w_mats'][:, src_indices[i]][None, :],
            'src_intrinsics': data_dict['intrinsics'][:, src_indices[i]][None, :],
            'src_masks_multi': data_dict['masks'][:, nearest_ids], # (1, 16, 128, 128, 1)
            'src_rgbs_multi': data_dict['rgbs'][:, nearest_ids], # (1, 16, 128, 128, 3)
            'src_c2w_mats_multi': data_dict['c2w_mats'][:, nearest_ids], # (1, 16, 4, 4)
            'src_w2c_mats_multi': data_dict['w2c_mats'][:, nearest_ids], # (1, 16, 4, 4)
            'src_intrinsics_multi': data_dict['intrinsics'][:, nearest_ids] # (1, 16, 4, 4)
        }

        samples.append(sample)

    return samples # len(samples) = 3, len(sample) = 17

def get_views_single(data_dict, src_indices, tgt_indices):
    """Acquire certain source/target views from a given sample
    
    Args:
        data_dict: sample from eval dataset
        src_indices: source view indices [#views]
        tgt_indices: target view indices [#views]
    
    Returns:
        An array of data_dict's
    """
    samples = []

    for i in range(len(src_indices)):
        sample = {
            'rgb_path': data_dict['rgb_path'],
            'img_id': data_dict['img_id'],
            'img_hw': data_dict['img_hw'],
            'depth_range': data_dict['depth_range'],
            'tgt_bbox': data_dict['bbox'][tgt_indices[i]],
            'tgt_mask': data_dict['masks'][tgt_indices[i]],
            'tgt_rgb': data_dict['rgbs'][tgt_indices[i]],
            'tgt_c2w_mat': data_dict['c2w_mats'][tgt_indices[i]],
            'tgt_intrinsic': data_dict['intrinsics'][tgt_indices[i]],
            'src_masks': data_dict['masks'][src_indices[i]][None, :], # HACK assuming one view
            'src_rgbs': data_dict['rgbs'][src_indices[i]][None, :],
            'src_c2w_mats': data_dict['c2w_mats'][src_indices[i]][None, :],
            'src_intrinsics': data_dict['intrinsics'][src_indices[i]][None, :],
        }

        samples.append(sample)

    return samples

def save_current_code(outdir):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = '.'
    dst_dir = os.path.join(outdir, 'code_{}'.format(date_time))
    shutil.copytree(src_dir, dst_dir,
                    ignore=shutil.ignore_patterns('data*', 'pretrained*', 'logs*', 'out*', '*.png', '*.mp4',
                                                  '*__pycache__*', '*.git*', '*.idea*', '*.zip', '*.jpg'))

# Get git commit hash
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    '''
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    '''
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2):
    '''
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    '''
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new


# tensor
def colorize(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False):
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 0.0,
    num_cycles: float = 0.999,
    last_epoch: int = -1,
):
    """
    https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/optimization.py#L129
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def natural_sort_key(s):
    parts = re.split('(\d+)', s)
    parts[0] = parts[0].lower()
    parts[1::2] = map(int, parts[1::2])
    return parts

def parse_pose(path):
    camera = np.load(path)

    intrinsics = np.zeros((4, 4))
    intrinsics[:3, :3] = camera['intr']
    intrinsics[3, 3] = 1.

    expr_array = camera['extr']
    R_c = expr_array[:, :3, :3].transpose(0, 2, 1)
    t_c = -R_c @ expr_array[:, :3, 3:]
    c2w_mats = np.concatenate((R_c, t_c), axis=2)

    ex_row = np.array([[0, 0, 0, 1]])
    ex_row = np.tile(ex_row, (expr_array.shape[0], 1, 1))
    w2c_mats = np.concatenate((expr_array, ex_row), axis=1)
    c2w_mats = np.concatenate((c2w_mats, ex_row), axis=1)

    return intrinsics, c2w_mats, w2c_mats

def ssim(rgb, gts):

    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    max_val = 1.0
    rgb = rgb.cpu().numpy()
    gts = gts.cpu().numpy()
    assert len(rgb.shape) == 3
    assert rgb.shape[-1] == 3
    assert rgb.shape == gts.shape
    import scipy.signal

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(rgb)
    mu1 = filt_fn(gts)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgb**2) - mu00
    sigma11 = filt_fn(gts**2) - mu11
    sigma01 = filt_fn(rgb * gts) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    return np.mean(ssim_map)

def render_image(args, model, ray_sampler, render_stride=1):
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        featmaps = model.encode(ray_batch['src_rgbs_multi'], ray_batch['src_intrinsics_multi'], ray_batch['src_c2w_mats_multi'])
        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  N_importance=args.N_importance,
                                  det=True,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)
    return ret

def msssim(rgb, gts):
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(torch.permute(rgb[None, ...], (0, 3, 1, 2)),
                   torch.permute(gts[None, ...], (0, 3, 1, 2))).item()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
