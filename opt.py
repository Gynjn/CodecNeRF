import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--ckptdir', type=str, help='checkpoint folder')
    parser.add_argument("--local-rank", type=int, default=0, help='rank for distributed training')   
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--iter', type=int, default=2000, help='iteration number')
    parser.add_argument('--data_type', type=str, default='obj', help='data type')

    ########## dataset options ##########
    ## train and eval dataset
    parser.add_argument('--data_path', type=str, help='the dataset to train')
    parser.add_argument('--camera_path', type=str, help='the camera data to train')
    parser.add_argument('--outdir', type=str, help='output directory')
    parser.add_argument('--img_hw', type=int, nargs='+', help='image size option for dataset')


    parser.add_argument('--N_rand', type=int, default=128,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='number of rays processed in parallel, decrease if running out of memory')

    ########### iterations & learning rate options ##########
    parser.add_argument('--lrate_mlp', type=float, default=1e-3, help='learning rate for mlp')
    parser.add_argument('--lrate_feat', type=float, default=5e-3, help='learning rate for feature map')
    parser.add_argument('--lrate_comp', type=float, default=1e-5, help='learning rate for compression model')

    ########## rendering options ##########
    parser.add_argument('--N_samples', type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument('--N_importance', type=int, default=64, help='number of important samples per ray')
    parser.add_argument('--inv_uniform', action='store_true',
                        help='if True, will uniformly sample inverse depths')
    parser.add_argument('--det', action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument('--white_bkgd', action='store_true',
                        help='apply the trick to avoid fitting to white background')
    parser.add_argument('--render_stride', type=int, default=1,
                        help='render with large stride for validation to save time')

    parser.add_argument('--i_print', type=int, default=100, help='frequency of terminal printout')
    parser.add_argument('--trank', type=int, default=4, help='rank for tensor decomposition')
    parser.add_argument('--lrank', type=int, default=4, help='rank for lora')
    parser.add_argument('--alpha', type=float, default=4.0, help='alpha for lora')

    return parser