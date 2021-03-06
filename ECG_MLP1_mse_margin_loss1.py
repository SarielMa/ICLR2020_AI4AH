import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from RobustDNN_loss import dZdX_loss1, multi_margin_loss
from Evaluate import test_adv
from ECG_Dataset import get_dataloader
from ECG_MLP1 import Net, main
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
#%%
def train(model, device, optimizer, dataloader, epoch, train_arg=None):
    model.train()#set model to training mode
    loss1_train=0
    loss2_train=0
    loss3_train=0
    acc_train =0
    sample_count=0
    M = torch.tensor([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,0],
                      [0,0,0,0,1]], device=device, dtype=torch.float32)
    if epoch > 0:
        model.initialize_dead_kernel()
    model.zero_WoW()
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #---------------------------
        model.zero_grad()
        model.normalize_kernel()
        Z = model(X)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y = Yp==Y
        #---------------------------
        MM=M[Y]
        loss1 = torch.mean(torch.sum((Z-MM)**2, dim=1))
        loss1.backward(retain_graph=True)
        loss2 = multi_margin_loss(Z[Yp_e_Y], Y[Yp_e_Y], margin=1, within_margin=True, reduction='counter')
        loss2.backward(retain_graph=True)
        loss3 = dZdX_loss1(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5)
        (0.2*loss3).backward()

        optimizer.step()

        model.update_WoW()

        loss1_train+=loss1.item()
        loss2_train+=loss2.item()
        loss3_train+=loss3.item()
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}'.format(
                  epoch, 100. * batch_idx / len(dataloader), loss1.item(), loss2.item(), loss3.item()))

    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)
    acc_train/=sample_count
    return (loss1_train, loss2_train, loss3_train), acc_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
#%%
    train_arg={}
    train_arg['lr']=0.001
    train_arg['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(batch_size=128,num=128, bias=False, loss_name='mse_margin_loss1',
         epoch_start=0, epoch_end=50, train=train, train_arg=train_arg, evaluate_model=True)