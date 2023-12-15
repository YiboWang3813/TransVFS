import os 
import time 
import json 
import random 
import datetime 
import argparse
from pathlib import Path 

import torch
import numpy as np 
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, BatchSampler  

import utils
from arguments import get_args_parser 
from image2force import build_dataset 
from models import build_model   
from engine import evaluate 


def main(args): 
    # initialize distributed setting here 
    utils.init_distributed_mode(args)  
    print("git:\n  {}\n".format(utils.get_sha())) 

    print(args) 

    device = torch.device(args.device) # set up device 
    
    # fix the seed for reproducibility 
    seed = args.seed + utils.get_rank() 
    torch.manual_seed(seed) 
    np.random.seed(seed) 
    random.seed(seed)
    
    model, criterion = build_model(args) # build model and criterion 
    model.to(device) 

    # set up the distributed model and print number of parameters 
    model_without_ddp = model 
    if args.distributed: 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) 
        model_without_ddp = model.module 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('number of parameters: {:.3f} MB'.format(num_parameters / 1e6))  

    # build test dataset, sampler, loader 
    dataset_val = build_dataset(args) 

    if args.distributed: 
        sampler_val = DistributedSampler(dataset_val, shuffle=False) 
    else: 
        sampler_val = SequentialSampler(dataset_val) 

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, 
                                 drop_last=False, num_workers=args.num_threads)
    
    output_dir = Path(args.output_dir) # set output file's path 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
        
    if args.resume: 
        checkpoint = torch.load(args.resume, map_location='cpu') 
        model_without_ddp.load_state_dict(checkpoint['model']) 

    if args.eval: # eval the model 
        test_stats = evaluate(model, criterion, data_loader_val, device, output_dir)  
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 
                     'num_parameters': num_parameters / 1e6}  
        if args.output_dir and utils.is_main_process(): 
            with open(output_dir / "log_eval.txt", 'a') as f: 
                f.write(json.dumps(log_stats) + '\n') 
        return 0 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser = get_args_parser(parser)  
    args = parser.parse_args() 
    main(args) 
