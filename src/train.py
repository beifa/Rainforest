import os
import time
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset, RandomSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score

from model import Res50
from dataset import RFDataset
from utils import lwlrap, visual_train_result

from torch.cuda import amp


SEED = 13
PATH_CSV = '../input'
PATH_NPY = '../input/train_npy'
LABEL = '../input/label.csv'
PATH_MODEL = '../models'
PATH_LOGS = '../logs'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

"""
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-f', '--fold', type=int, help = 'add fold number', default=0)    
    parser.add_argument('-e', '--epoch', type=int, help = 'add count epoch', default= 10)    
    parser.add_argument('-n', '--num_workers', type=int, help = 'num_workers', default= 4)
    parser.add_argument('-m', '--model', type=str, help = 'name model to train : res50, eff(name, out)', default = 'res50')
    parser.add_argument('-l', '--loss_func', type=str, help = 'loss func : BCEWithLogitsLoss, BCELoss, dice_loss, FocalLoss, MixedLoss ', default = 'BCEWithLogitsLoss')
    parser.add_argument('-d', '--debag', type=bool, help = 'small data set', default = False)
    


    pars = parser.parse_args()

"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, required=True, help = 'name_model_size_epoch_typefold_scheduler_mixadd')
    parser.add_argument('--fold_type', type=str, required=True, help = 'version fold')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--epoch', type=int, default = 30)
    parser.add_argument('--n_workers', type=int, default = 4)
    parser.add_argument('--batch', type=int, default = 14)
    parser.add_argument('--size', type=int, default = 128) 
    parser.add_argument('--scaler', action='store_false', help = 'AUTOMATIC MIXED PRECISION')        
    args, _ = parser.parse_known_args()
    return args

def  train(model, loader, loss_f, optimizer, scaler):
    model.train()
    bar = tqdm(loader)
    train_loss = []
    train_target = []
    train_prob = []
    for (img, target) in bar:
        optimizer.zero_grad()
        if args.scaler:
            # Runs the forward pass with autocasting.
            with amp.autocast():
                img, target = img.to(device), target.to(device)
                y_ = model(img)
                pred = loss_f(y_, target)
                scaler.scale(pred).backward()
                scaler.step(optimizer)
                scaler.update()            
        else:
            img, target = img.to(device), target.to(device)
            y_ = model(img)
            pred = loss_f(y_, target)
            pred.backward()
            optimizer.step()
        
        train_loss.append(pred.detach().cpu().numpy())
        prob = nn.Sigmoid()(y_)        
        train_prob.append(prob.detach().cpu())
        train_target.append(target.detach().cpu())            
    
    probs =  torch.cat(train_prob).numpy()
    target = torch.cat(train_target).numpy()   
    auc = roc_auc_score(target, probs)
    return train_loss, auc


def val_train(model, loader, loss_f):
    val_loss = []
    target_temp = []
    probs = []
    model.eval()
    with torch.no_grad():
        bar = tqdm(loader)
        for (img, target) in bar:
            img, target = img.to(device), target.to(device)
            y_ = model(img) 
            
            pred = loss_f(y_, target)            
            prob = nn.Sigmoid()(y_)  
            
            val_loss.append(pred.detach().cpu())        
            probs.append(prob.detach().cpu())
            target_temp.append(target.detach().cpu())   
            
    val_loss = np.mean(val_loss) 
    probs =  torch.cat(probs).numpy()
    target = torch.cat(target_temp).numpy()
   
    auc = roc_auc_score(target, probs)
    lraps = label_ranking_average_precision_score(target, probs)

    score_class, weight = lwlrap(target, probs)
    score_loss = (score_class * weight).sum()

    return val_loss, auc, lraps, score_loss


def showtime(model, f: int, df: pd.DataFrame, label:pd.DataFrame, scaler):
    start = time.ctime().replace('  ', ' ').replace(' ', '_')    
    print('Fold: ', f)
    if args.DEBUG:
        print('DEBUG....................')
        start = 'DEBUG_' + start
        args.epoch = 5
        args.size = 64        
        print(args)
        tr_data = df[df.sfold !=f].sample(200).reset_index(drop=True)
        vl_data = df[df.sfold ==f].sample(200).reset_index(drop=True)
    else:
        tr_data = df[df.sfold !=f].reset_index(drop=True)
        vl_data = df[df.sfold ==f].reset_index(drop=True)
    
    tr_dataset = RFDataset(PATH_NPY, label.loc[tr_data.index], size = args.size)
    vl_dataset = RFDataset(PATH_NPY, label.loc[vl_data.index], size = args.size)
    
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch, num_workers=args.n_workers,
                           sampler=RandomSampler(tr_dataset))
    vl_loader = DataLoader(vl_dataset, batch_size=args.batch, num_workers=args.n_workers)    

    # kernel_type = type(model).__name__
    model = model.to(device)
    # need addd pos_weight 
    pos_weights = torch.ones(24)
    pos_weights = pos_weights * 24    
    loss_f = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weights).to(device)       
    optimizer = optim.Adam(model.parameters(), lr = 3e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)


    auc_max = 0
    for ep in range(args.epoch):
        print('Epoch: ', ep + 1)
        
        train_loss, auc_train = train(model, tr_loader, loss_f, optimizer, scaler)
        val_loss, auc, lraps, score_loss = val_train(model, vl_loader, loss_f)

        log = time.ctime().replace('  ', ' ').replace(' ', '_') + ',' + f'Fold:{f},Epoch:{ep},lr:{optimizer.param_groups[0]["lr"]:.5f},Auc_val:{auc:.5f},Auc_train:{auc_train:.5f},LRAPS:{lraps:.5f},lwlrap:{score_loss:.5f},train_loss:{np.mean(train_loss):.5f},val_loss:{val_loss:.5f}'
        print(log)
        to_log = start + '.' + f'log_{args.kernel}.txt'          
        with open(os.path.join(PATH_LOGS, to_log), 'a') as file:
            file.write(log + '\n')

        if auc > auc_max:
            print(f'auc_max: {auc} --> {auc_max}). Saving model ...')
            torch.save(model.state_dict(), os.path.join(PATH_MODEL, f'{args.kernel}_best_fold_{f}.pth'))
            auc_max = auc       
        
        scheduler.step(val_loss) 
    visual_train_result(to_log)   
    torch.save(model.state_dict(), os.path.join(PATH_MODEL, f'{args.kernel}_final_fold_{f}.pth'))
    torch.cuda.empty_cache()   

if __name__ == "__main__":
    """
     

    params = {
        'SEED': 13,
        'batch_size': 32,
        'lr': 1e-4,
        'num_workers' : pars.num_workers,
        'epoch': pars.epoch,
        'fold': pars.fold,
        'model': pars.model,
        'loss_func' : pars.loss_func
    }
    
    """

    df = pd.read_csv(os.path.join(PATH_CSV, 'gfold_sfold_df.csv')) # two way folds
    label = pd.read_csv(os.path.join(PATH_CSV, 'label.csv')) # two way folds
    set_seed(SEED)
    args = parse_args()
    scaler = amp.GradScaler()  
    model = Res50()
    fold = 0
    showtime(model, fold, df, label, scaler)
