import os
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm
import imageio
from dataset.obj_dataset import OBJDataset
from models.render_ray import render_rays
from models.render_image import render_single_image
from models.model import CodecNerfModel
from models.sample_ray import RaySamplerSingleImage, RaySamplerMultipleImages
from models.criterion import Criterion
from entropy_coding.base import SimpleVAECompressionModel
from entropy_coding.weight_entropy import *
import opt
import math
from utils import (
    img2mse, mse2psnr, img_HWC2CHW, img2psnr, set_seed,
    get_cosine_schedule_with_warmup, ssim, msssim)
import pandas as pd

torch.set_float32_matmul_precision('high')


def finetune(args):
    device = "cuda:{}".format(args.local_rank)
    dataset = OBJDataset(args.data_path, args.camera_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = CodecNerfModel(args)
    compress_model = SimpleVAECompressionModel(args).to(device)
    spike_and_slap_cdf = SpikeAndSlabCDF()
    weight_entropyModule = WeightEntropyModule(spike_and_slap_cdf)
    weight_entropyModule.to(device)
    weight_entropyModule.train()    
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
        compress_param = {
            "net": {
                name for name, param in compress_model.named_parameters()
                if param.requires_grad and not name.endswith(".quantiles")
            },
        }
        compress_param = {k: [p for p in compress_model.parameters() if p.requires_grad] for k, v in compress_param.items()}
        vm_model = torch.nn.ParameterList(feat_coefs).to(device)
        vm_model.train()
        params = model.get_param()
        params += [{'params': vm_model, 'lr':args.lrate_feat}]
        params += [{'params': compress_param['net'], 'lr': args.lrate_comp}]
        optim = torch.optim.Adam(params, eps=1e-15)

        lr_sched = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=100, num_training_steps=args.iter)

        for i in range(args.iter):

            m0 = vm_model[0]
            m1 = vm_model[2]
            m0_h, likelihood0 = compress_model(m0)
            m1_h, likelihood1 = compress_model(m1)
            m_0_ = m0_h.view(3, args.trank, -1)
            m_1_ = m1_h.view(3, args.trank, -1)

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

            num_pixels = args.img_hw[0] * args.img_hw[1]
            bpp_loss0 = torch.log(likelihood0).sum() / (-math.log(2) * num_pixels)
            bpp_loss1 = torch.log(likelihood1).sum() / (-math.log(2) * num_pixels)
            loss += 1e-5 * (bpp_loss0 + bpp_loss1)
            w_bpp_sum = 0
            for key in model.net_coarse.state_dict():
                if 'lora' in key:
                    weight = model.net_coarse.state_dict()[key] 
                    og_shape = weight.shape
                    weight = weight.reshape(1,1,-1)

                    w_hat, w_likelihood = weight_entropyModule(weight, True)
                    w_hat = w_hat.reshape(og_shape)

                    w_bpp = torch.log(w_likelihood) / (-math.log(2) * num_pixels)
                    w_bpp_sum += w_bpp.sum()                        

            for key in model.net_fine.state_dict():
                if 'lora' in key:
                    weight = model.net_fine.state_dict()[key] 
                    og_shape = weight.shape
                    weight = weight.reshape(1,1,-1)

                    w_hat, w_likelihood = weight_entropyModule(weight, True)
                    w_hat = w_hat.reshape(og_shape)

                    w_bpp = torch.log(w_likelihood) / (-math.log(2) * num_pixels)
                    w_bpp_sum += w_bpp.sum()                   
            loss += 1e-5 * w_bpp_sum            
            loss.backward()
            scalars_to_log['loss'] = loss.item()
            optim.step()
            lr_sched.step()
            compress_model.update(force=True)

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
                compress_model.eval()
                weight_entropyModule.eval()

                m0 = vm_model[0]
                m1 = vm_model[2]
                string0 = compress_model.entropy_bottleneck.compress(m0)
                string1 = compress_model.entropy_bottleneck.compress(m1)
                shape0 = m0.size()[2:]
                shape1 = m1.size()[2:]
                
                m0_hat = compress_model.entropy_bottleneck.decompress(string0, shape0)
                m1_hat = compress_model.entropy_bottleneck.decompress(string1, shape1)

                m_0_ = m0_hat.view(3, args.trank, -1)
                m_1_ = m1_hat.view(3, args.trank, -1)
                v_0_ = vm_model[1]
                v_1_ = vm_model[3]

                vm_merged0 = torch.einsum('ijl,jk->ikl', m_0_, v_0_)
                vm_merged1 = torch.einsum('ijl,jk->ikl', m_1_, v_1_)

                vm_merged0 = vm_merged0.view(3, 32, feat_res_list[0], feat_res_list[0]).unsqueeze(1)
                vm_merged1 = vm_merged1.view(3, 32, feat_res_list[1], feat_res_list[1]).unsqueeze(1)

                vm_merged0 += to_optimize0
                vm_merged1 += to_optimize1

                string_mb = (len(string0[0]) + len(string0[1]) + len(string0[2]) +\
                            len(string1[0]) + len(string1[1]) + len(string1[2]))/(10**6)
                
                bit_sum = 0
                state_dict_coarse = model.net_coarse.state_dict()
                for key in model.net_coarse.state_dict():
                    if 'lora' in key:  
                        weight = model.net_coarse.state_dict()[key] 
                        og_shape = weight.shape
                        weight = weight.reshape(1,1,-1)

                        bits = weight_entropyModule.compress(weight)
                        bit_sum += len(bits[0])
                        w_hat = weight_entropyModule.decompress(bits, og_shape)
                        w_hat = w_hat.reshape(og_shape)
                        state_dict_coarse[key] = w_hat.cpu()                        
                model.net_coarse.load_state_dict(state_dict_coarse, strict=False)

                state_dict_fine = model.net_fine.state_dict()
                for key in model.net_fine.state_dict():
                    if 'lora' in key:  
                        weight = model.net_fine.state_dict()[key] 
                        og_shape = weight.shape
                        weight = weight.reshape(1,1,-1)

                        bits = weight_entropyModule.compress(weight)
                        bit_sum += len(bits[0])
                        w_hat = weight_entropyModule.decompress(bits, og_shape)
                        w_hat = w_hat.reshape(og_shape)
                        state_dict_fine[key] = w_hat.cpu()                        
                model.net_fine.load_state_dict(state_dict_fine, strict=False)                
                bit_sum = bit_sum/(10**6)

                storage = bit_sum + string_mb + (args.trank * 32 * 2 * 4)/(10**6) # vector size

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
                                    "CODE" : code_mb, "REST": storage}}
                new_df = pd.DataFrame([(folder, metrics['ITER'], metrics['PSNR'], metrics['SSIM'], metrics['MSSSIM'],
                                        metrics['CODE'], metrics['REST']) for folder, metrics in pd_data.items()],
                                        columns=['FileName', 'ITER', 'PSNR', 'SSIM', 'MSSSIM', 'CODE', 'REST'])
                existing_csv_file_path = f"{i}_ec_weight.csv"

                if os.path.exists(existing_csv_file_path):
                    new_df.to_csv(existing_csv_file_path, index=False, mode='a', header=False)
                    
                    print(f"Data appended to the existing CSV file at: {existing_csv_file_path}")
                else:
                    new_df.to_csv(existing_csv_file_path, index=False)                    

                vm_model.train()
                model.switch_to_train()
                compress_model.train()
                weight_entropyModule.train()


if __name__ == '__main__':
    parser = opt.config_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    finetune(args)
