import torch
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from model import VAE, VAE_FC, vae_loss
from util import PngFolder, auc, setup_seed, forward
from tqdm import tqdm
import torch.nn as nn
import random
import pdb
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import pickle

setup_seed(99)

import pdb
pdb.set_trace()

with open("/workshop/16QAM_Train_Test.pkl", "rb") as f:
    data = pickle.load(f)

train_data = data["train_data"]
train_label = data["train_label"]
test_data = data["test_data"]
test_label = data["test_label"]

# setup data
train_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(train_data, train_label),
        batch_size=64,
        shuffle=True
)
test_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(test_data, test_label),
        batch_size=64,
        shuffle=True
)

# setup model
vae_model = VAE().cuda()
vae_model.load_state_dict(torch.load("/workshop/combine-nopool-maxpoolmask-attn8/mae_best.pth"))

# feature learning
with torch.no_grad():

    summary = []
    for data, label in test_loader:
        (output, mean, logvar), data = forward(vae_model, data)
        summary.append([data, label, output, mean, logvar]) 

    summary = zip(*summary)
    summary = map(
            lambda x: torch.cat(x, dim=0), 
            list(summary)
    )
    data, label, output, mean, logvar = list(summary)
    
    save_image(data[label==0][-64:], "/result/input-test-n.png")
    save_image(output[label==0][-64:], "/result/output-test-n.png")
    save_image(data[label==1][-64:], "/result/input-test-a.png")
    save_image(output[label==1][-64:], "/result/output-test-a.png")

    loss = vae_loss(output, mean, logvar, data)
    mae_auc, mse_auc, xujing_auc, attn_auc = auc(data, label, output)

    print("[Val] loss:{}, auc:{}|{}|{}|{}".format(
        loss, mae_auc, mse_auc, xujing_auc, attn_auc
    ))

# feature extraction and labeling
X = torch.cat([mean, logvar], dim=1).cpu().numpy()
label = label.cpu().numpy()

# lof detector
lof = LocalOutlierFactor(n_neighbors=500)
_ = lof.fit_predict(X)
lof_score = lof.negative_outlier_factor_

# svm detector
#svm = OneClassSVM(gamma='auto').fit(X)
svm = OneClassSVM(gamma='scale').fit(X)
_ = svm.fit_predict(X)
svm_score = svm.score_samples(X)

# EllipticEnvelope detector
cov = EllipticEnvelope(random_state=0).fit(X)
cov_score = cov.decision_function(X)

iso = IsolationForest(random_state=0).fit(X)
iso_score = iso.score_samples(X)

# score
lof_auc = roc_auc_score(label, -lof_score)
svm_auc = roc_auc_score(label, -svm_score)
cov_auc = roc_auc_score(label, -cov_score)
iso_auc = roc_auc_score(label, -iso_score)
print(lof_auc, svm_auc, cov_auc, iso_auc)
