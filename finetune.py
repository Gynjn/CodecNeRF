import os
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm
import imageio
from dataset.obj_dataset import OBJDataset
from dataset.gso_dataset import GSODataset
from models.render_ray import render_rays
from models.render_image import render_single_image
from models.model import CodecNerfModel
from models.sample_ray import RaySamplerSingleImage, RaySamplerMultipleImages
from models.criterion import Criterion
import opt
from utils import (
    img2mse, mse2psnr, img_HWC2CHW, img2psnr, set_seed,
    get_cosine_schedule_with_warmup, ssim, msssim)
import pandas as pd

torch.set_float32_matmul_precision('high')


def finetune(args):
    device = "cuda:{}".format(args.local_rank)
    if args.data_type == 'obj':
        dataset = OBJDataset(args.data_path, args.camera_path)
    elif args.data_type == 'gso':
        dataset = GSODataset(args.data_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = CodecNerfModel(args)
    criterion = Criterion()
    scalars_to_log = {}

    for train_data in data_loader:
        ray_samplert = RaySamplerMultipleImages(train_data, device, args.img_hw)

        time0 = time.time()
    
        featmaps, code_mb = model.encode(train_data['src_rgbs'], train_data['src_intrinsics'], None, train_data['src_w2c_mats'])

        to_optimize0 = nn.Parameter(featmaps[0], requires_grad=False).to(device)
        to_optimize1 = nn.Parameter(featmaps[1], requires_grad=False).to(device)

        n_feature_maps = len(featmaps)
        feat_dim_list = [featmaps[0].shape[2], featmaps[1].shape[2]]
        feat_res_list = [featmaps[0].shape[-1], featmaps[1].shape[-1]]

        del featmaps, model.feature_net

        feat_coefs = []
        for i in range(n_feature_maps):
            m = nn.Parameter(torch.randn((3, args.trank, feat_res_list[i], feat_res_list[i])))
            v = nn.Parameter(torch.randn((args.trank, feat_dim_list[i])))
            feat_coefs.append(nn.init.normal_(m))
            feat_coefs.append(nn.init.zeros_(v))           

        vm_model = torch.nn.ParameterList(feat_coefs).to(device)
        vm_model.train()
        params = model.get_param()
        params += [{'params': vm_model, 'lr':args.lrate_feat}]
        optim = torch.optim.Adam(params, eps=1e-15)

        lr_sched = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=100, num_training_steps=args.iter)

        for i in range(args.iter+1):

            m_0_ = vm_model[0].view(3, args.trank, -1)
            m_1_ = vm_model[2].view(3, args.trank, -1)

            v_0_ = vm_model[1]
            v_1_ = vm_model[3]

            vm_merged0 = torch.einsum('ijl,jk->ikl', m_0_, v_0_)
            vm_merged1 = torch.einsum('ijl,jk->ikl', m_1_, v_1_)

            vm_merged0 = vm_merged0.view(3, 32, feat_res_list[0], feat_res_list[0]).unsqueeze(1)
            vm_merged1 = vm_merged1.view(3, 32, feat_res_list[1], feat_res_list[1]).unsqueeze(1)

            vm_merged0 += to_optimize0
            vm_merged1 += to_optimize1


            ray_batch = ray_samplert.random_sample(args.N_rand)
            ret = render_rays(ray_batch=ray_batch,
                            model=model,
                            featmaps=[vm_merged0, vm_merged1],
                            N_samples=args.N_samples,
                            inv_uniform=args.inv_uniform,
                            N_importance=args.N_importance,
                            det=args.det,
                            white_bkgd=args.white_bkgd,
                            )
        
            optim.zero_grad()
            loss, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)                        

            if ret['outputs_fine'] is not None:
                fine_loss, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                loss += fine_loss
            
            loss.backward()
            scalars_to_log['loss'] = loss.item()
            optim.step()
            lr_sched.step()

            dt = time.time() - time0            

            if args.local_rank == 0:
                if i % args.i_print == 0 or i < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                        scalars_to_log['train/fine-loss'] = mse_error
                        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)
                    try:
                        logstr = '{} step: {} '.format(train_data['scene_name'][0], i)
                    except:
                        logstr = '{} step: {} '.format(args.data_path.split('/')[-1], i)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                    print(logstr)
                    print('each iter time {:.05f} seconds'.format(dt))

            if i % 1000 == 0:
                vm_model.eval()
                model.switch_to_eval()                                  

                m_0_ = vm_model[0].view(3, args.trank, -1)
                m_1_ = vm_model[2].view(3, args.trank, -1)

                v_0_ = vm_model[1]
                v_1_ = vm_model[3]

                vm_merged0 = torch.einsum('ijl,jk->ikl', m_0_, v_0_)
                vm_merged1 = torch.einsum('ijl,jk->ikl', m_1_, v_1_)

                vm_merged0 = vm_merged0.view(3, 32, feat_res_list[0], feat_res_list[0]).unsqueeze(1)
                vm_merged1 = vm_merged1.view(3, 32, feat_res_list[1], feat_res_list[1]).unsqueeze(1)

                vm_merged0 += to_optimize0
                vm_merged1 += to_optimize1

                print('Logging a random validation view...')
                try:
                    ren_folder = os.path.join(args.outdir, train_data['scene_name'][0], str(i))
                except:
                    ren_folder = os.path.join(args.outdir, args.data_path.split('/')[-1], str(i))
                print(f'Rendering to {ren_folder}')
                os.makedirs(ren_folder, exist_ok=True)

                data_input = dict(
                    tgt_intrinsic=train_data['all_intrinsics'][0][0:1],
                    depth_range=train_data['depth_range'],
                )

                
                render_poses = train_data['all_c2w_mats'][0]
                view_indices = np.arange(0, len(render_poses), 1)

                psnr_ = 0
                ssim_ = 0
                msssim_ = 0
                with torch.no_grad():

                    for idx, pose in tqdm.tqdm(zip(view_indices, render_poses), total=len(view_indices)):
                        filename = os.path.join(ren_folder, f'{idx:06}.png')
                        data_input['tgt_c2w_mat'] = pose[None, :]                        
                        ray_sampler = RaySamplerSingleImage(data_input, device, args.img_hw, render_stride=1)
                        ray_batch = ray_sampler.get_all()

                        ret = render_single_image(ray_sampler=ray_sampler,
                                                ray_batch=ray_batch,
                                                model=model,
                                                chunk_size=args.chunk_size,
                                                N_samples=args.N_samples,
                                                inv_uniform=args.inv_uniform,
                                                N_importance=args.N_importance,
                                                det=True,
                                                white_bkgd=args.white_bkgd,
                                                render_stride=1,
                                                featmaps=[vm_merged0, vm_merged1])
                        check_ = ret['outputs_fine']['rgb']
                        check_ = torch.clamp(check_.detach().cpu(), 0.0, 1.0)
                        psnr_ += img2psnr(check_, train_data['all_rgbs'][0][idx])
                        ssim_ += ssim(check_, train_data['all_rgbs'][0][idx])
                        msssim_ += msssim(check_, train_data['all_rgbs'][0][idx])
                        rgb_im = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
                        rgb_im = torch.clamp(rgb_im, 0.0, 1.0)
                        rgb_im = rgb_im.permute([1, 2, 0]).cpu().numpy()

                        rgb_im = (rgb_im * 255.).astype(np.uint8)
                        imageio.imwrite(filename, rgb_im)
                
                try:
                    name = train_data['scene_name'][0]
                except:
                    name = args.data_path.split('/')[-1]
                pd_data = {name : {'ITER': i, 'PSNR' : psnr_/len(render_poses), 'SSIM' : ssim_/len(render_poses), 
                                    'MSSSIM' : msssim_/len(render_poses),
                                    "CODE" : code_mb}}
                new_df = pd.DataFrame([(folder, metrics['ITER'], metrics['PSNR'], metrics['SSIM'], metrics['MSSSIM'],
                                        metrics['CODE']) for folder, metrics in pd_data.items()],
                                        columns=['FileName', 'ITER', 'PSNR', 'SSIM', 'MSSSIM', 'CODE'])      
                existing_csv_file_path = f"{i}_ft.csv"

                if os.path.exists(existing_csv_file_path):
                    new_df.to_csv(existing_csv_file_path, index=False, mode='a', header=False)
                    
                    print(f"Data appended to the existing CSV file at: {existing_csv_file_path}")
                else:
                    new_df.to_csv(existing_csv_file_path, index=False)                    

                vm_model.train()
                model.switch_to_train()


if __name__ == '__main__':
    parser = opt.config_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    finetune(args)
