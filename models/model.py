import os
import torch
from torch import nn
from network.codec import Codec
import tinycudann as tcnn
import loralib as lora


class CodecNerfModel(object):
    def __init__(self, args):
        self.args = args
        device = torch.device('cuda:{}'.format(args.local_rank))

        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 2,
            },
        )    
                         
        self.feature_net = Codec(input_dim_2d=3, extractor_channel=64, 
                                 input_dim_3d=16*16, dims_3d=(32, 64, 128, 256), 
                                 input_dim_tri=32, 
                                 feat_dim_tri=64, 
                                 spatial_volume_length=1.0, 
                                 V = 64, device=device).to(device)

        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.start_step = self.load_from_ckpt(args.ckptdir)

        self.net_coarse = nn.Sequential(
            lora.Linear(80, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 4, args.lrank, args.alpha, bias=False),
        ).to(device)
        lora.mark_only_lora_as_trainable(self.net_coarse)

        self.net_fine = nn.Sequential(
            lora.Linear(80, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 64, args.lrank, args.alpha, bias=False),
            nn.ReLU(),
            lora.Linear(64, 4, args.lrank, args.alpha, bias=False),
        ).to(device)
        lora.mark_only_lora_as_trainable(self.net_fine)

        nn_ckpt = torch.load("checkpoints/nn.pth")
        self.net_coarse.load_state_dict(nn_ckpt["nn_coarse"], strict=False)
        self.net_fine.load_state_dict(nn_ckpt["nn_fine"], strict=False)

    def encode(self, x, K, poses, w2cs):
        """
        Args:
            x: input tensor [b, v, h, w, c]
            K: intrinsics [b, v, 4, 4]
            poses: extrinsics [b, v, 4, 4]
        Returns:
            Extracted feature maps [b, out_c, h, w]
        """
        feat_maps, _, code_mb, _ = self.feature_net(x, K, poses, w2cs)

        return feat_maps, code_mb

    def switch_to_eval(self):
        self.net_coarse.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        lora.mark_only_lora_as_trainable(self.net_coarse)                            
        if self.net_fine is not None:
            self.net_fine.train()
            lora.mark_only_lora_as_trainable(self.net_fine)  

    def load_model(self, filename):
        to_load = torch.load(filename)
        self.feature_net.load_state_dict(to_load['feature_net'])

        # self.net_coarse.load_state_dict(to_load['net_coarse'])
        # if self.net_fine is not None and 'net_fine' in to_load.keys():
        #     self.net_fine.load_state_dict(to_load['net_fine'])


    def load_from_ckpt(self, out_folder):
        '''Load model from existing checkpoints and return the current step
        
        Args:
            out_folder: the directory that stores ckpts
        Returns:
            The current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        fpath = ckpts[0]
        self.load_model(fpath)
        step = int(fpath[-10:-4])
        print('Reloading from {}, starting at step={}'.format(fpath, step))
        step = 0

        return step

    def get_param(self):
        params = [
            {'params': self.net_coarse.parameters(), 'lr': self.args.lrate_mlp},
            {'params': self.net_fine.parameters(), 'lr': self.args.lrate_mlp},
        ]
        return params