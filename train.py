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

setup_seed(99)

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
#vae_model.load_state_dict(torch.load("png-flip-pow-seperate-shufflenobn/attn_best.pth"))

# setup tensorboard
writer = SummaryWriter("/result")

# setup optimizer
# TODO use l2 reg
optimizer = Adam(vae_model.parameters(), lr=1e-4)

# log best auc
mae_best = 0

# start training
for epoch in range(1000):

    # train
    for idx, (data, label) in enumerate(train_loader):

        # forward
        (output, mean, logvar), data = forward(vae_model, data)
        loss = vae_loss(output, mean, logvar, data)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch:{} [{}/{}], loss:{}".format(
            epoch, idx, len(train_loader), loss
        ))

        step = idx + len(train_loader) * epoch
        writer.add_scalars("loss", {"train": loss.item()}, step)

    save_image(data, "/result/input-train.png")
    save_image(output, "/result/output-train.png")

    # validate
    if epoch % 5 == 4:
        
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
        
            print("[Val] Epoch:{}, loss:{}, auc:{}|{}|{}|{}".format(
                epoch, loss, mae_auc, mse_auc, xujing_auc, attn_auc
            ))
            
            writer.add_scalars("loss", {"test": loss.item()}, step)
            writer.add_scalars("auc", {"mae": mae_auc.item()}, step)
            writer.add_scalars("auc", {"attn": attn_auc.item()}, step)
            writer.add_scalars("auc", {"xujing": xujing_auc.item()}, step)
            writer.add_scalars("auc", {"mse": mse_auc.item()}, step)

        if mae_auc > mae_best:
            mae_best = mae_auc
            torch.save(vae_model.state_dict(), "/result/mae_best.pth") 
