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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import label_ranking_average_precision_score

from model import Res50
from dataset import RFDataset
from utils import lwlrap, visual_train_result, FocalLoss

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

def accuracy(pred: np.array, target: np.array)->np.array:    
    # add same idea by accuracy metric
    corrects = 0
    tt, tar_idx = torch.max(target, dim= 1)
    aa, ans_idx = torch.max(pred, dim= 1) #dim=0 values, if dim =1 return max value and index max value
    # print('tr-->', tt, tar_idx)
    # print('as-->', aa,ans_idx)
    
    for i in range(0, len(ans_idx)):
        if ans_idx[i] == tar_idx[i]:
            corrects = corrects + 1  
    return corrects

def fuck_auc(target:np.array, probs: np.array)-> float:
     # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
    try:
        auc = roc_auc_score(target, probs) 
    except ValueError:
        auc = 0
    return auc

def  train(model, loader, loss_f, optimizer, scaler):
    model.train()
    bar = tqdm(loader)
    train_loss = []
    train_target = []
    train_prob = []
    correct = []
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
        correct.append(accuracy(y_, target))    
        train_prob.append(prob.detach().cpu())
        train_target.append(target.detach().cpu())            
    
    probs =  torch.cat(train_prob).numpy()
    target = torch.cat(train_target).numpy()
    auc = fuck_auc(target, probs)
    print(f'TR Correct : {sum(correct)} / {len(target)}')
    return train_loss, auc


def val_train(model, loader, loss_f):
    val_loss = []
    target_temp = []
    probs = []
    correct = []
    model.eval()
    with torch.no_grad():
        bar = tqdm(loader)
        for (img, target) in bar:
            img, target = img.to(device), target.to(device)
            y_ = model(img) 
            
            pred = loss_f(y_, target)            
            prob = nn.Sigmoid()(y_)  

            correct.append(accuracy(y_, target)) 

            val_loss.append(pred.detach().cpu())        
            probs.append(prob.detach().cpu())
            target_temp.append(target.detach().cpu())   
            
    val_loss = np.mean(val_loss) 
    probs =  torch.cat(probs).numpy()
    target = torch.cat(target_temp).numpy()
   
    auc = fuck_auc(target, probs) 

    pr = average_precision_score(target, probs, average='weighted')
    pr_m = average_precision_score(target, probs, average='micro')

    lraps = label_ranking_average_precision_score(target, probs)

    score_class, weight = lwlrap(target, probs)
    score_loss = (score_class * weight).sum()
    print(f'VL Correct : {sum(correct)} / {len(target)}')
    return val_loss, auc, lraps, score_loss, pr, pr_m, sum(correct)


def showtime(model, f: int, df: pd.DataFrame, label:pd.DataFrame, scaler):
    start = time.ctime().replace('  ', ' ').replace(' ', '_')    
    print('Fold: ', f)
    tr_idx = df[args.fold_type] != f
    vl_idx = df[args.fold_type] == f

    if args.DEBUG:
        print('DEBUG....................')
        start = 'DEBUG_' + start
        args.epoch = 1               
        print(args)        
        tr_label = label.loc[tr_idx].sample(200).reset_index(drop=True)
        vl_label = label.loc[vl_idx].sample(200).reset_index(drop=True)
    else:
        tr_label = label.loc[tr_idx].reset_index(drop=True)
        vl_label = label.loc[vl_idx].reset_index(drop=True)
    print(tr_label.shape, vl_label.shape)
    tr_dataset = RFDataset(PATH_NPY, tr_label, size = args.size)
    vl_dataset = RFDataset(PATH_NPY, vl_label, size = args.size)
    
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch, num_workers=args.n_workers,
                           sampler=RandomSampler(tr_dataset))
    vl_loader = DataLoader(vl_dataset, batch_size=args.batch, num_workers=args.n_workers)    

    # kernel_type = type(model).__name__
    model = model.to(device)
    # need addd pos_weight 
    pos_weights = torch.ones(24)
    pos_weights = pos_weights * 24    
    loss_f = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weights).to(device)
    # loss_f = FocalLoss()       
    optimizer = optim.Adam(model.parameters())#, lr = 0.001, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9, mode='min', patience=4, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=3, verbose=True)

    auc_max = 0
    correct_max = 0    
    for ep in range(args.epoch):
        print('Epoch: ', ep + 1)
        
        train_loss, auc_train = train(model, tr_loader, loss_f, optimizer, scaler)
        val_loss, auc, lraps, score_loss, pr,pr_m, correct = val_train(model, vl_loader, loss_f)

        log = time.ctime().replace('  ', ' ').replace(' ', '_') + ',' + f'Fold:{f},Epoch:{ep},lr:{optimizer.param_groups[0]["lr"]:.7f},Auc_val:{auc:.5f},Auc_train:{auc_train:.5f},LRAPS:{lraps:.5f},lwlrap:{score_loss:.5f},train_loss:{np.mean(train_loss):.5f},val_loss:{val_loss:.5f}'
               
        print(log)
        to_log = start + '.' + f'log_{args.kernel}.txt'          
        with open(os.path.join(PATH_LOGS, to_log), 'a') as file:
            file.write(log + '\n')

        if auc > auc_max:
            print(f'auc_max: {auc} --> {auc_max}, PRw: {pr}, PRmicro: {pr_m}). Saving model ...')
            torch.save(model.state_dict(), os.path.join(PATH_MODEL, f'{args.kernel}_best_fold_{f}.pth'))
            auc_max = auc      
        
        scheduler.step(val_loss)         
    visual_train_result(to_log)   
    torch.save(model.state_dict(), os.path.join(PATH_MODEL, f'{args.kernel}_final_fold_{f}.pth'))
    torch.cuda.empty_cache()   

if __name__ == "__main__":

    tp = pd.read_csv(os.path.join(PATH_CSV, 'train_tp.csv'))
    fp = pd.read_csv(os.path.join(PATH_CSV, 'train_fp.csv'))
    df = pd.concat((tp, fp))
    df = df.sample(frac = 1).reset_index(drop=True)
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    
    df['kfold'] = -1
    for f, (tr, vl) in enumerate(skf.split(df, df.species_id)):
        print(len(tr), len(vl))
        df.loc[vl, 'kfold'] = f  

    # df = pd.read_csv(os.path.join(PATH_CSV, 'gfold_sfold_df.csv')) # two way folds   
    label = pd.read_csv(os.path.join(PATH_CSV, 'label.csv')) # two way folds
    # label = pd.read_csv(os.path.join(PATH_CSV, 'label_tp.csv'))
    set_seed(SEED)
    args = parse_args()
    scaler = amp.GradScaler()  
    model = Res50()
    fold = 0
    showtime(model, fold, df, label, scaler)