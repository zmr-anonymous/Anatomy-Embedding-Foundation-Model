from monai.losses import DiceCELoss, DiceFocalLoss
from utility import *

from loss.soft_dice_ce_loss import soft_DC_and_CE_loss
from loss.SAM.SAM_loss import SAM_loss
from loss.SAM.SAM_loss_unet import SAM_loss_unet
from loss.SAM.SAM_loss_moco import SAM_loss_moco
from loss.SAM.SAM_loss_moco_fed import SAM_loss_moco_fed
from loss.SAM.SAM_loss_moco_contrastive import SAM_loss_moco_contrastive
from loss.SAM.SAM_loss_moco_ce import SAM_loss_moco_ce
from loss.SAM.SAM_loss_confidence import SAM_loss_confidence
from loss.SAM.SAM_loss_moco_cluster import SAM_loss_moco_cluster

def get_loss(config:dict):
    loss_name = config['Train']['loss']
    if loss_name == 'DiceCELoss':
        return DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, softmax=True)
    elif loss_name == 'DiceFocalLoss':
        return DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, softmax=True)
    elif loss_name == 'soft_DC_and_CE_loss':
        batch_dice = config['Train']['dataloader_geodesic']['batch_size']
        return soft_DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
    elif loss_name == 'SAM_loss':
        return SAM_loss(config)
    elif loss_name == 'SAM_loss_unet':
        return SAM_loss_unet(config)
    elif loss_name == 'SAM_loss_moco':
        tau = config['Train']['SAM_loss_moco']['tau']
        device = globalVal.device
        if 'only_local' in config['Train']['SAM_loss_moco'].keys() and config['Train']['SAM_loss_moco']['only_local']:
            return SAM_loss_moco(tau, device, only_local=True)
        else:
            return SAM_loss_moco(tau, device)
    elif loss_name == 'SAM_loss_moco_fed':
        tau = config['Train']['SAM_loss_moco']['tau']
        device = globalVal.device
        return SAM_loss_moco_fed(tau, device)
    elif loss_name == 'SAM_loss_moco_contrastive':
        tau = config['Train']['SAM_loss_moco_contrastive']['tau']
        device = globalVal.device
        return SAM_loss_moco_contrastive(tau, device)
    elif loss_name == 'SAM_loss_moco_ce':
        tau = config['Train']['SAM_loss_moco_ce']['tau']
        device = globalVal.device
        return SAM_loss_moco_ce(tau, device)
    elif loss_name == 'SAM_loss_confidence':
        device = globalVal.device
        if 'only_local' in config['Train']['SAM_loss_confidence'].keys() and config['Train']['SAM_loss_confidence']['only_local']:
            return SAM_loss_confidence(device, only_local=True)
        else:
            return SAM_loss_confidence(device)
    elif loss_name == 'SAM_loss_moco_cluster':
        tau = config['Train']['SAM_loss_moco_cluster']['tau']
        n_cluster = config['Train']['SAM_loss_moco_cluster']['n_cluster']
        device = globalVal.device
        if 'only_local' in config['Train']['SAM_loss_moco_cluster'].keys() and config['Train']['SAM_loss_moco_cluster']['only_local']:
            return SAM_loss_moco_cluster(tau, n_cluster, device, only_local=True)
        else:
            return SAM_loss_moco_cluster(tau, n_cluster, device)


        