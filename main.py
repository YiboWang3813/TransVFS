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
from engine import train_one_epoch, evaluate 


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
    print('number of parameters: {:.6f} MB'.format(num_parameters / 1e6))  

    # initialize optimizer and lr scheduler 
    optimizer = torch.optim.Adam(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    # build train/test(val) dataset, sampler, loader 
    dataset_train = build_dataset(args, 'train') 
    dataset_val = build_dataset(args, 'eval') 

    if args.distributed: 
        sampler_train = DistributedSampler(dataset_train, shuffle=False)  
        sampler_val = DistributedSampler(dataset_val, shuffle=False) 
    else: 
        sampler_train = RandomSampler(dataset_train) 
        sampler_val = SequentialSampler(dataset_val) 
    
    batch_sampler_train = BatchSampler(sampler_train, args.batch_size, drop_last=True) 
     
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, 
                                   num_workers=args.num_threads) 
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, 
                                 drop_last=False, num_workers=args.num_threads)
    
    output_dir = Path(args.output_dir) # set output file's path 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
        
    # load trained weight to model (eval), also load trained weight to optimizer, lr scheduler (train) and reset start epoch 
    if args.resume: 
        checkpoint = torch.load(args.resume, map_location='cpu') 
        model_without_ddp.load_state_dict(checkpoint['model']) 
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint: 
        #     optimizer.load_state_dict(checkpoint['optimizer']) 
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 
        #     args.start_epoch = checkpoint['epoch'] + 1 

    if args.eval: # eval the model 
        test_stats = evaluate(model, criterion, data_loader_val, device, output_dir)  
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 
                     'num_parameters': num_parameters / 1e6}  
        if args.output_dir and utils.is_main_process(): 
            with open(output_dir / "log_eval.txt", 'a') as f: 
                f.write(json.dumps(log_stats) + '\n') 
        return 0 
    
    print('Start training') 
    start_time = time.time() # train the model 
    for epoch in range(args.start_epoch, args.n_epochs): 
        # train model one epoch and get trained status, adjust learning rate 
        if args.distributed: 
            sampler_train.set_epoch(epoch) 
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, 
                                      device, epoch, args.clip_max_norm) 
        lr_scheduler.step() 

        # save latest checkpoint and certain epoch's checkpoint 
        if args.output_dir: 
            checkpoint_paths = [output_dir / 'checkpoint.pth'] 
            # extract checkpoint before LR drop and every 1 epochs 
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0: 
                checkpoint_paths.append(output_dir / f'checkpoint{(epoch + 1):04}.pth') 
            for checkpoint_path in checkpoint_paths: 
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(), 
                    # 'optimizer': optimizer.state_dict(), 
                    # 'lr_scheduler': lr_scheduler.state_dict(), 
                    'epoch': epoch, 
                    'args': args}, checkpoint_path) 
        
        # eval the model after train and get tested status 
        test_stats = evaluate(model, criterion, data_loader_val, device, output_dir) 

        # combine trained and tested status together and save them to local txt file 
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, 
                     'num_parameters': num_parameters / 1e6}  
        
        if args.output_dir and utils.is_main_process(): 
            with open(output_dir / 'log_train.txt', 'a') as f: 
                f.write(json.dumps(log_stats) + '\n') 

    total_time = time.time() - start_time 
    total_time_str = str(datetime.timedelta(seconds=int(total_time))) 
    print('Training time: {}'.format(total_time_str))  
                

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser = get_args_parser(parser)  
    args = parser.parse_args() 
    main(args) 
