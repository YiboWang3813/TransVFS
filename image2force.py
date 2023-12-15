import os
import json

import torch 
import nibabel 
import numpy as np 
import scipy.io as io 
from torch.utils.data.dataset import Dataset 

import transforms 
import random 

from typing import List 


def build_dataset(args, set='train'): 
    args.root_dir = os.path.join(args.dataroot, set) 
    args.set = set 
    # with open('settings_mask.json', 'r') as f: 
    #     dir_mask = json.load(f) 
    # f.close() 
    # args.angles_pool = dir_mask[args.data_source][args.image_type][set] 
    dataset = ImagetoForceDataset(args) 
    return dataset 


class ImagetoForceDataset(Dataset): 
    """ Define a dataset class to load 4d spatio-temporal prostate's volume sequence. 
    
    Parameters: 
        root_dir (str): the root directory where stores data 
        num_steps (int): number of time steps in a batch of data 
        num_futures (int): number of future steps will be applied 
        transforms (Dict): whether to apply data agumentation 
        
    Returns: 
        input_images (Tensor): input volume sequence [B, N, C, D, H, W] 
        output_force (Tensor): output force vector [B, Nf] 
        where B denotes batch size, N denotes number of time steps, C denotes input channels like 1, 
        D, H, W denote depth, height, width of input image respectively, and Nf denotes number of forces. """
    def __init__(self, args): 
        super().__init__() 

        angle_names = os.listdir(args.root_dir) 
        angle_names = sorted_angle_dir(angle_names) 

        image_paths, force_paths, box_paths = [], [], [] 
        num_files_per_dir = [] 
        for angle_name in angle_names: 
            # skip some bad angle dir if needed to improve accuracy 
            # if angle_name not in args.angles_pool: 
            #     continue 

            # set angle dir
            angle_dir = os.path.join(args.root_dir, angle_name) 
            # set image dir and load images in this angle dir 
            image_dir = os.path.join(angle_dir, 'image')  
            image_names = os.listdir(image_dir) 
            image_names = sorted_listdir(image_names) 
            for image_name in image_names: 
                image_paths.append(os.path.join(image_dir, image_name)) 
            # set force dir and load forces in this angle dir 
            force_dir = os.path.join(angle_dir, 'force') 
            force_names = os.listdir(force_dir) 
            force_names = sorted_listdir(force_names) 
            for force_name in force_names: 
                force_paths.append(os.path.join(force_dir, force_name)) 

            # log number of files (image, force, box) in each angle dir 
            # making sure each batch only contains files in one angle dir 
            num_files_per_dir.append(len(image_names)) 

            if args.is_box: 
                # set box dir and load boxes in this angle dir when working in boxing mode 
                box_dir = os.path.join(angle_dir, 'box') 
                box_names = os.listdir(box_dir) 
                box_names = sorted_listdir(box_names) 
                for box_name in box_names: 
                    box_paths.append(os.path.join(box_dir, box_name)) 

        assert len(image_paths) == len(force_paths), "The number of images should be equal to the number of forces"

        self.image_paths = image_paths 
        self.force_paths = force_paths 
        
        # compute accumulated files preparing for computing relative index 
        self.num_files_accumulated = [] 
        for i in range(len(num_files_per_dir)): 
            self.num_files_accumulated.append(sum(num_files_per_dir[:i+1]))
        self.num_files_accumulated = [0] + [int(n) for n in self.num_files_accumulated]

        self.num_steps = args.num_steps 
        self.num_futures = args.num_futures 
        self.is_box = False 
        self.is_noise = False 

        if args.is_box:  
            assert len(image_paths) == len(box_paths), "The number of images should be equal to the number of boxes" 
            self.box_paths = box_paths 
            self.is_box = True 
            self.cp_shape = tuple([int(cs) for cs in args.cp_shape.split(',')])

        if args.is_noise: 
            self.is_noise = True 
            self.noise_mode = args.noise_mode 
            self.noise_var = args.noise_var 

    def __len__(self): 
        return len(self.image_paths)   

    def _get_input_idx(self, index): 
        absolute_index = None 
        num_files = None 

        # find index involved in which dir 
        for i in range(len(self.num_files_accumulated) - 1): 
            start_point = self.num_files_accumulated[i] 
            end_point = self.num_files_accumulated[i+1] 
            if start_point <= index <= end_point: 
                absolute_index = start_point 
                num_files = end_point - start_point 

        relative_index = index - absolute_index 

        # get input sequence indexs according to relative input index 
        seq_inds = None 
        if relative_index < self.num_steps - 1: # num_steps = 4 e.g 0,1,2
            seq_inds = [relative_index] * self.num_steps 
        elif relative_index > num_files - self.num_futures - 1: # num_futures = 1 
            seq_inds = [relative_index - self.num_futures + j for j in list(range(-self.num_steps + 1, 1))]
        else: 
            seq_inds = [relative_index + j for j in list(range(-self.num_steps + 1, 1))] 

        seq_inds = [idx + absolute_index for idx in seq_inds] 

        return seq_inds  

    def _load_image(self, index): 
        image_path = self.image_paths[index] 
        image = nibabel.load(image_path).dataobj 
        image = np.array(image, dtype='float32') 
        if self.is_box: 
            box_path = self.box_paths[index] 
            with open(box_path, 'r') as f: 
                box = json.load(f) 
            f.close() 
            image = transforms.crop(image, box['box']) 
            image = transforms.pad(image, self.cp_shape) 
        if self.is_noise: 
            image = transforms.noise(image, self.noise_mode, self.noise_var) 
        image = transforms.normalize(image) 
        image = np.array(image, dtype='float32')
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(1) # add time and channel axis 
        return image 
    
    def _load_force(self, index): 
        force_path = self.force_paths[index] 
        force = io.loadmat(force_path)['force'] 
        force = np.array(force, dtype='float32')
        force = torch.from_numpy(force).reshape(-1) 
        return force 

    def __getitem__(self, index): 
        input_idxs = self._get_input_idx(index) 
        output_idx = input_idxs[-1] + self.num_futures 

        input_images = [] 
        for in_idx in input_idxs: 
            image = self._load_image(in_idx)  
            input_images.append(image) 
        input_images = torch.cat(input_images, dim=0) # [L, 1, D, H, W] 

        force = self._load_force(output_idx) # [6] 
        output_force = force 

        return input_images, output_force 

    def __str__(self): 
        return "Image2ForceDataset"


def sorted_angle_dir(angle_names: List): 
    indices = sorted(
        range(len(angle_names)), 
        key=lambda index: int(angle_names[index].split('e')[-1]) 
    ) 
    sorted_angle_names = [angle_names[index] for index in indices] 
    return sorted_angle_names 


def sorted_listdir(list1: List): 
    """ sort file names according to the integral order """
    indices = sorted(
        range(len(list1)), 
        key=lambda index: int(list1[index].split('.')[0]) 
    )
    list2 = [list1[index] for index in indices] 
    return list2 