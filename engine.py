import os 
import sys 
import math 

import torch 
import pandas as pd 
import matplotlib.pyplot as plt 

import utils 

from typing import Iterable 


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0): 
    model.train() 
    criterion.train() 
    metric_logger = utils.MetricLogger(delimeter="  ") 
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}')) 
    header = 'Epoch: [{}]'.format(epoch) 
    print_freq = 1  

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header): 
        samples = samples.to(device) 
        targets = targets.to(device) 

        outputs = model(samples) 
        loss_dict = criterion(outputs, targets) 
        weight_dict = criterion.weight_dict 
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) 

        # reduce losses over all GPUs for logging purposes 
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value): 
            print('Loss is {}, stop training'.format(loss_value)) 
            print(loss_dict_reduced) 
            sys.exit(1) 

        optimizer.zero_grad() 
        losses.backward() 
        # if max_norm > 0: 
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 
        optimizer.step() 

        metric_logger.update(lr=optimizer.param_groups[0]['lr'])  
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled) 
        
    metric_logger.synchronize_between_processes() 
    print('Averaged stats:', metric_logger) 
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  


@torch.no_grad() 
def evaluate(model, criterion, data_loader, device, output_dir):  
    model.eval() 
    criterion.eval() 
    metric_logger = utils.MetricLogger(delimeter="  ") 
    header = 'Test:' 
    print_freq = 1 
    force_true = {'Fx': [], 'Fy': [], 'Fz': [], 'Tx': [], 'Ty': [], 'Tz': []} 
    force_pred = {'Fx': [], 'Fy': [], 'Fz': [], 'Tx': [], 'Ty': [], 'Tz': []} 

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header): 
        samples = samples.to(device) 
        targets = targets.to(device) 

        outputs = model(samples) 
        loss_dict = criterion(outputs, targets) 
        weight_dict = criterion.weight_dict 

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()} 
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values()) 
        loss_value = losses_reduced_scaled.item() 
        
        # log metrics, predicted and grount truth force 
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled) 
    
        for b in range(outputs.shape[0]): 
            fx, fy, fz, tx, ty, tz = outputs[b].cpu().numpy() 
            force_pred['Fx'].append(fx) 
            force_pred['Fy'].append(fy) 
            force_pred['Fz'].append(fz) 
            force_pred['Tx'].append(tx) 
            force_pred['Ty'].append(ty) 
            force_pred['Tz'].append(tz) 
            
            fx, fy, fz, tx, ty, tz = targets[b].cpu().numpy() 
            force_true['Fx'].append(fx) 
            force_true['Fy'].append(fy) 
            force_true['Fz'].append(fz) 
            force_true['Tx'].append(tx) 
            force_true['Ty'].append(ty) 
            force_true['Tz'].append(tz) 

    # synchronize results between all processes and save force dataframe to local disk 
    metric_logger.synchronize_between_processes() 
    print('Averaged stats:', metric_logger) 

    # save predicted and true forces in to .xlsx file 
    force_df_pred = pd.DataFrame(force_pred) 
    force_df_pred.to_excel(os.path.join(output_dir, 'force_pred.xlsx'), index=False) 
    force_df_true = pd.DataFrame(force_true) 
    force_df_true.to_excel(os.path.join(output_dir, 'force_true.xlsx'), index=False) 

    # plot predicted and true forces in one .png file 
    titles = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'] 
    plt.figure(figsize=(24, 36)) 
    for i in range(len(titles)): 
        plt.subplot(6, 1, i + 1) 
        plt.plot(force_pred[titles[i]], color='b', label='pred', ls='--') 
        plt.plot(force_true[titles[i]], color='r', label='true', ls='-')   
        plt.legend() 
        plt.title(titles[i]) 
    plt.savefig(os.path.join(output_dir, 'force.png')) 

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  
