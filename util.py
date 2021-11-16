import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import DatasetFolder, ImageFolder
import torchvision.transforms as transforms
from functools import partial
import scipy.io as scio
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torchvision.utils import save_image


def forward(model, data):
    
    data = data.cuda().float()
    data = data * 10

    return model(data), data


def auc(x, label, y):

    diff = x - y
    
    mae = diff.abs().sum(dim=[1,2,3]).detach().cpu()
    mae_auc = roc_auc_score(label, -mae)
    
    mse = diff.abs().pow(2).sum(dim=[1,2,3]).detach().cpu()
    mse_auc = roc_auc_score(label, -mse)

    # TODO adjust thres, kernel, lambda
    
    xujing = (diff / x).abs().sum(dim=[1,2,3]).detach().cpu()
    xujing_auc = roc_auc_score(label, -xujing)
    
    thres = 0.05
    kernel = 3
    mask = torch.nn.functional.max_pool2d(
            x, kernel_size=kernel, stride=1, padding=kernel//2)
    mask = mask < thres
    
    pool = torch.nn.functional.max_pool2d(
            -diff.abs(), kernel_size=kernel, stride=1, padding=kernel//2)
    pool = -pool
    #pool = torch.nn.functional.max_pool2d(
    #        diff.abs(), kernel_size=kernel, stride=1, padding=kernel//2)

    bg_mask = mask
    attn1 = (pool*bg_mask).flatten(start_dim=1).cpu()
    attn1 = np.percentile(attn1, 90, axis=1)
    
    sig_mask = ~mask
    attn2 = (pool*sig_mask).flatten(start_dim=1).cpu()
    attn2 = np.percentile(attn2, 99, axis=1)

    attn = 2*attn1 + attn2
    attn_auc = roc_auc_score(label, -attn)

    return mae_auc, mse_auc, xujing_auc, attn_auc


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
