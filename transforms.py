import torch 
import numpy as np 

from skimage.util import random_noise 


def normalize(x: np.ndarray, mode: str = 'mm'): 
    if mode == 'mm': 
        _min, _max = np.min(x), np.max(x) 
        _x = (x - _min) / (_max - _min) 
    elif mode == 'ms': 
        _mean, _std = np.mean(x), np.std(x) 
        _x = (x - _mean) / _std 
    else: 
        raise NotImplementedError("normalization {} has not been implemented".format(mode)) 
    
    return _x 


def crop(x: np.ndarray, crop_box: dict): 
    zmin, zmax = crop_box['zmin'], crop_box['zmax'] 
    ymin, ymax = crop_box['ymin'], crop_box['ymax'] 
    xmin, xmax = crop_box['xmin'], crop_box['xmax'] 

    return x[xmin:xmax, ymin:ymax, zmin:zmax] 


def pad(x: np.ndarray, pad_shape: tuple): 
    old_width, old_height, old_depth = x.shape 
    new_width, new_height, new_depth = pad_shape 
    
    # adjust width 
    if new_width > old_width: 
        left_half_width = (new_width - old_width) // 2 
        right_half_width = new_width - old_width - left_half_width 
        x = np.concatenate([np.zeros((left_half_width, old_height, old_depth)), x], axis=0) 
        x = np.concatenate([x, np.zeros((right_half_width, old_height, old_depth))], axis=0) 
    elif new_width == old_width: 
        pass 
    else: 
        left_half_width = (old_width - new_width) // 2 
        right_half_width = old_width - new_width - left_half_width 
        x = x[left_half_width:-right_half_width, :, :] 
    old_width = new_width 

    # adjust height 
    if new_height > old_height: 
        top_half_height = (new_height - old_height) // 2 
        bottom_half_height = new_height - old_height - top_half_height 
        x = np.concatenate([np.zeros((old_width, top_half_height, old_depth)), x], axis=1) 
        x = np.concatenate([x, np.zeros((old_width, bottom_half_height, old_depth))], axis=1) 
    elif new_height == old_height: 
        pass 
    else: 
        top_half_height = (old_height - new_height) // 2 
        bottom_half_height = old_height - new_height - top_half_height 
        x = x[:, top_half_height:-bottom_half_height, :] 
    old_height = new_height 

    # adjust depth 
    if new_depth > old_depth: 
        front_half_depth = (new_depth - old_depth) // 2 
        end_half_depth = new_depth - old_depth - front_half_depth 
        x = np.concatenate([np.zeros((old_width, old_height, front_half_depth)), x], axis=2) 
        x = np.concatenate([x, np.zeros((old_width, old_height, end_half_depth))], axis=2) 
    elif new_depth == old_depth: 
        pass 
    else: 
        front_half_depth = (old_depth - new_depth) // 2 
        end_half_depth = old_depth - new_depth - front_half_depth 
        x = x[:, :, front_half_depth:-end_half_depth]  
    old_depth = new_depth  

    return x 


def noise(x: np.ndarray, mode: str, var: float): 
    
    def add_noise_to_slice(slice, mode, var): 
        if mode == 'gaussian': 
            noised_slice = random_noise(slice, 'gaussian', var=var) 
        elif mode == 's&p': 
            noised_slice = random_noise(slice, 's&p', amount=var) 
        elif mode == 'speckle': 
            noised_slice = random_noise(slice, 'speckle', var=var) 
        else: 
            raise NotImplementedError("{} noise mode has not been implemented".format(mode)) 
        return noised_slice 

    for d in range(x.shape[-1]): 
        slice = x[:, :, d] 
        noised_slice = add_noise_to_slice(slice, mode, var) 
        x[:, :, d] = noised_slice 

    return x 
