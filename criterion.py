import torch 
import torch.nn as nn 
from torch.nn import init 

from typing import Dict, List 


class SetCriterion(nn.Module): 
    """ Define a criterion module to calculate losses and metrics. 
    
    Parameters: 
        weight_dict (Dict): a dict to control each loss's weight, which should be specified with each loss 
        losses (List): a list contains all loss or metric name needed to computed """

    __annotations__ = {
        "weight_dict": {'loss_mse_f': 1, 'loss_mse_t': 0.1, 'loss_mse': 1, 'loss_l1': 1}, 
        "losses": ['mse_f', 'mse_t', 'mse', 'mae', 'pcc', 'l1']
    } # the key name in weight dict should be as same as the key name in returned loss value dict 

    def __init__(self, weight_dict, losses): 
        # type: (Dict, List) -> None 
        super().__init__() 

        self.weight_dict = weight_dict 
        self.losses = losses 

        self.mse_fn = nn.MSELoss() 
        self.l1_fn = nn.L1Loss() 

    def loss_mse(self, outputs, targets): 
        """ Compute Mean Square Error. """ 
        loss_mse = self.mse_fn(outputs, targets) 
        return {'loss_mse': loss_mse} 

    def loss_mse_force(self, outputs, targets): 
        """ Compute Mean Square Error only for forces. """ 
        loss_mse = self.mse_fn(outputs[:, 0:3], targets[:, 0:3]) 
        return {'loss_mse_f': loss_mse} 

    def loss_mse_torque(self, outputs, targets): 
        """ Compute Mean Square Error only for torques. """ 
        loss_mse = self.mse_fn(outputs[:, 3:6], targets[:, 3:6]) 
        return {'loss_mse_t': loss_mse} 

    def loss_l1(self, outputs, targets): 
        """ Compute Mean Absolute Error (L1). """ 
        loss_l1 = self.l1_fn(outputs, targets) 
        return {'loss_l1': loss_l1}

    @torch.no_grad() 
    def loss_mae(self, y_pred, y_true): 
        """ Compute Mean Absolute Error. This is not a real loss. It doesn't propagate gradients. """
        y_diff = torch.abs(y_pred - y_true) 
        # compute mean absolute error 
        force_diff = y_diff[:, :3] 
        torque_diff = y_diff[:, 3:] 
        force_mae_mean = torch.mean(force_diff) 
        force_mae_std = torch.std(force_diff) 
        torque_mae_mean = torch.mean(torque_diff) 
        torque_mae_std = torch.std(torque_diff) 
        # mae_mean = force_mae_mean + torque_mae_mean 
        # mae_std = force_mae_std + torque_mae_std 
        # compute relative mean absolute error 
        force_true = y_true[:, :3] 
        torque_true = y_true[:, 3:] 
        force_true_std = torch.std(force_true) 
        torque_true_std = torch.std(torque_true) 
        force_rmae_mean = force_mae_mean / force_true_std 
        force_rmae_std = force_mae_std / force_true_std 
        torque_rmae_mean = torque_mae_mean / torque_true_std 
        torque_rmae_std = torque_mae_std / torque_true_std 
        # rmae_mean = force_rmae_mean + torque_rmae_mean 
        # rmae_std = force_rmae_std + torque_rmae_std 
        losses = {} 
        losses.update({'force_mae_mean': force_mae_mean, 'force_mae_std': force_mae_std}) 
        losses.update({'torque_mae_mean': torque_mae_mean, 'torque_mae_std': torque_mae_std}) 
        losses.update({'force_rmae_mean': force_rmae_mean, 'force_rmae_std': force_rmae_std}) 
        losses.update({'torque_rmae_mean': torque_rmae_mean, 'torque_rmae_std': torque_rmae_std})
        # return 
        # losses = {'mae_mean': mae_mean, 'mae_std': mae_std, 'rmae_mean': rmae_mean, 'rmae_std': rmae_std} 
        return losses 
    
    @torch.no_grad() 
    def loss_pcc(self, y_pred, y_true): 
        """ Compute Person Correlation Coefficient error. This is not a real loss. It doesn't propagate gradients. """
        y_pred = y_pred.reshape(-1) 
        y_true = y_true.reshape(-1)
        y_pred_mean = torch.mean(y_pred) 
        y_true_mean = torch.mean(y_true) 
        top = torch.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) 
        bottom = torch.sqrt(torch.sum((y_pred - y_pred_mean) ** 2)) * \
                torch.sqrt(torch.sum((y_true - y_true_mean) ** 2))
        pcc = top / bottom 
        losses = {'pcc': pcc} 
        return losses 
    
    def get_loss(self, loss, outputs, targets, **kwargs): 
        """ Getting loss according to specific loss name. """
        loss_map = {
            'mse': self.loss_mse, 
            'mae': self.loss_mae, 
            'pcc': self.loss_pcc, 
            'mse_f': self.loss_mse_force, 
            'mse_t': self.loss_mse_torque, 
            'l1': self.loss_l1}
        assert loss in loss_map, f"do you really want to compute {loss} loss ?" 
        return loss_map[loss](outputs, targets, **kwargs)  

    def forward(self, outputs, targets): 
        """ Forward function. 

        Parameters: 
            outputs (Tensor): network's predicted forces [B, C] 
            targets (Tensor): ground truth forces [B, C] 
        
        Returns: 
            losses (Dict): losses and accuracy dict 
        where B denotes batch size, C denotes number of forces 6 = [Fx, Fy, Fz, Tx, Ty, Tz] """
        losses = {} 
        for loss in self.losses: 
            losses.update(self.get_loss(loss, outputs, targets))
        
        return losses 


# initialize network's weights copied from CycleGAN, we thank for Mr. Zhu's great contributions to CV 
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>