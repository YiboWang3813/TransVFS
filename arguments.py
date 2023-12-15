import argparse


def add_dataset_related_arguments(parser: argparse.ArgumentParser): 
    parser.add_argument('--dataroot', type=str, default='dataset/Prostate4D', help='dataset root dir') 
    parser.add_argument('--data_source', type=str, default='Phantom', help='the source of data') 
    parser.add_argument('--image_type', type=str, default='Rectum', help='the imaging type of data') 
    parser.add_argument('--pp', type=str, default='up', help='up or down, pull or push')
    parser.add_argument('--batch_size', type=int, default=1, help='number of images in a batch') 
    parser.add_argument('--num_steps', type=int, default=4, help='number of time steps in a batch') 
    parser.add_argument('--num_futures', type=int, default=0, help='number of future time steps applied to load force') 
    parser.add_argument('--num_threads', type=int, default=4, help='number of threads used to load data') 
    parser.add_argument('--vol_shape', type=str, default='128,128,128', help='input volume shape')
    parser.add_argument('--is_box', action='store_true') 
    parser.add_argument('--cp_shape', type=str, default='64,64,64', help='the dst shape after cropped and padded')
    parser.add_argument('--is_noise', action='store_true') 
    parser.add_argument('--noise_mode', type=str, default='speckle', help='the mode of noise')
    parser.add_argument('--noise_var', type=float, default=0., help='the varience of noise') 
    return parser 

def add_helper_related_arguments(parser: argparse.ArgumentParser): 
    parser.add_argument('--device', type=str, default='cuda', help='device for training or testing')
    parser.add_argument('--seed', type=int, default=42, help='random seed to make sure results reproducable') 
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay coefficient') 
    parser.add_argument('--lr_drop', type=int, default=5, help='same to step size in lr scheduler') 
    parser.add_argument('--output_dir', type=str, default='', help='output directory stores weight') 
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoints') 
    parser.add_argument('--eval', action='store_true') # if you set --eval, then args.eval is True 
    parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to start training') 
    parser.add_argument('--n_epochs', type=int, default=15, help='number of epochs for training')
    parser.add_argument('--clip_max_norm', type=float, default=0.1, help='gradient clipping max norm')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed process') 
    parser.add_argument('--local_rank', default=0, type=int, help='rank of this python program') 
    parser.add_argument('--dist_url', default='env://', help='url to set up distributed training') 
    return parser 

def add_common_arguments(parser: argparse.ArgumentParser): 
    parser.add_argument('--in_channels', type=int, default=1, help='input channels') 
    parser.add_argument('--out_channels', type=int, default=6, help='output channels') 
    parser.add_argument('--net_name', type=str, default='', help='which network will be trained or tested') 
    parser.add_argument('--depths', type=str, default='Base', help='which type of network blocks you want to use | Small | Base | Large') 
    parser.add_argument('--num_heads', type=str, default='1,2,4,8', help='number of multi heads in attention') 
    parser.add_argument('--num_folds', type=int, default=10, help='number of folds to carry out cross validation') 
    return parser 

def add_transvfs_related_arguments(parser: argparse.ArgumentParser): 
    parser.add_argument('--mode', type=str, default='C', help='which type of spatio-temporal attention you want to use A | B | C | D | E | F ') 
    parser.add_argument('--lambda_', type=float, default=1, help='lambda scale for balancing force and torque weight')
    parser.add_argument('--gap', type=str, default='Simple', help='which type of gap you want to use Simple') 
    parser.add_argument('--channels', type=str, default='8,16,32,64', help='the number of input channels list to build stfet') 
    parser.add_argument('--ssd', type=str, default='4,2,2,2', help='spatial strides to reduce spatial resolution') 
    parser.add_argument('--tsd', type=str, default='1,1,2,1', help='temporal strides to reduce temporal resolution') 
    return parser 

def get_args_parser(parser: argparse.ArgumentParser): 
    parser = add_dataset_related_arguments(parser) 
    parser = add_helper_related_arguments(parser) 
    parser = add_common_arguments(parser) 
    parser = add_transvfs_related_arguments(parser) 
    return parser 
