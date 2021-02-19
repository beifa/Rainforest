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

from model import EBlite4, EBLite4_260, EBLite4_384
from dataset import RFDataset
from utils import lwlrap, visual_train_result, GradualWarmupSchedulerV2

from torch.cuda import amp


SEED = 13
PATH_CSV = '../input'
PATH_NPY = ''
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
    parser.add_argument('--PARAM', type=int, required=True, help = 'choice version dataset 260 or 384')
    parser.add_argument('--EXP', dest = 'EXP', action = 'store_true', help = 'choice Exp data')
    parser.add_argument('--NO-EXP', dest = 'EXP', action = 'store_false', help = 'default data')
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


def showtime(model, f: int, data, tr_idx: np.array, vl_idx: np.array, scaler, rd):

    start = time.ctime().replace('  ', ' ').replace(' ', '_')    
    print('Fold: ', f)

    if args.PARAM == 260:        
        # 260
        tr = np.take(data.files[1:], tr_idx[f])
        vl = np.take(data.files[1:], vl_idx[f])
        ver = 'v2'
    else:
        # 384
        tr = np.take(data.file_name.values, tr_idx[f])
        vl = np.take(data.file_name.values, vl_idx[f]) 
        ver = 'v1'
    print(tr.shape, vl.shape)


    if args.DEBUG:
        df = pd.read_csv('../input/train_tp.csv')
        print('DEBUG....................used 384')
        start = 'DEBUG_' + start
        args.epoch = 1               
        print(args)        
        tr = np.take(df.file_name.values, tr_idx[f])
        vl = np.take(df.file_name.values, vl_idx[f]) 
        ver = 'v1'
    
    # version v1 - 384, v2 - 260   
    tr_dataset = RFDataset(tr, PATH_ZIP, version= ver, size = None, rand = rd)
    vl_dataset = RFDataset(vl, PATH_ZIP, version= ver, size = None, rand = rd)

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch, num_workers=args.n_workers,
                           sampler=RandomSampler(tr_dataset))
    vl_loader = DataLoader(vl_dataset, batch_size=args.batch, num_workers=args.n_workers)    

    # kernel_type = type(model).__name__
    model = model.to(device)
    # need addd pos_weight 
    pos_weights = torch.ones(24)
    pos_weights = pos_weights * 24    
    loss_f = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weights).to(device)
    # loss_f = FocalLoss(), bad 

    # default    
    # optimizer = optim.Adam(model.parameters())#, lr = 0.001, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=3, verbose=True)    
    
    """ 
    sub2_BLite4_sr32power2mel384_notchageSize_folddata_addlayers_RAND_best_None.csv, [0.882, 0.880]
    sub2_BLite4_sr48power2mel260_notchageSize_folddata_addlayers_RAND_best_None.csv, [0.862, 0.860]
    """


    #by optune, kaggle score 0.882, change seed 0.888
    optimizer = optim.Adam(model.parameters(),
                           lr = 0.001597,
                          weight_decay = 0.000216)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)
    scheduler = GradualWarmupSchedulerV2(optimizer,
                                         multiplier=1, 
                                         total_epoch=5,
                                         after_scheduler=scheduler_cosine)
    """
    sub2_BLite4_sr32power2mel384_RAND_schedulerGWS_CosineALR_best_None.csv, [0.879, 0.873]
    sub2_BLite4_sr48power2mel260_RAND_schedulerGWS_CosineALR_best_None.csv, [0.877, 0.871]    
    """

    lraps_max = 0
    auc_max = 0
    loss = np.inf 
    correct_max = 0    
    for ep in range(args.epoch):
        print('Epoch: ', ep + 1)
        
        train_loss, auc_train = train(model, tr_loader, loss_f, optimizer, scaler)
        val_loss, auc, lraps, score_loss, pr,pr_m, correct = val_train(model, vl_loader, loss_f)
        scheduler.step()

        log = time.ctime().replace('  ', ' ').replace(' ', '_') + ',' + f'Fold:{f},Epoch:{ep},lr:{optimizer.param_groups[0]["lr"]:.7f},Auc_val:{auc:.5f},Auc_train:{auc_train:.5f},LRAPS:{lraps:.5f},lwlrap:{score_loss:.5f},train_loss:{np.mean(train_loss):.5f},val_loss:{val_loss:.5f}'
               
        print(log)
        to_log = start + '.' + f'log_{args.kernel}.txt'          
        with open(os.path.join(PATH_LOGS, to_log), 'a') as file:
            file.write(log + '\n')

        # if auc > auc_max:
        if (lraps > lraps_max) and (val_loss < loss) :
            print(f'lraps_max: {lraps} --> {lraps_max}, PRw: {pr}, PRmicro: {pr_m}). Saving model ...')
            torch.save(model.state_dict(), os.path.join(PATH_MODEL, f'{args.kernel}_best_fold_{f}.pth'))
            lraps_max = lraps
            loss = val_loss      
        
        # if use plateau
        # scheduler.step(val_loss)         
    visual_train_result(to_log)   
    torch.save(model.state_dict(), os.path.join(PATH_MODEL, f'{args.kernel}_final_fold_{f}.pth'))
    torch.cuda.empty_cache()   

if __name__ == "__main__":  
    from sklearn.model_selection import StratifiedKFold
    tr_idx = []
    vl_idx = []
    set_seed(SEED)
    args = parse_args()
    scaler = amp.GradScaler()     
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # IF USE RAND NEED CHANGE DATA data 10 sec
    if args.EXP:
        rand_data = True        
        if args.PARAM == 260:            
            PATH_ZIP = '../input/exp_make_img_sr48power2mel260/train_img.zip' 
        else:
            PATH_ZIP = '../input/exp_make_img_sr32power2mel384_ff111/train_img.zip'
    else:  
        rand_data = None      
        if args.PARAM == 260:            
            PATH_ZIP = '../input/sr48power2mel260/train_img.zip'
        else:
            PATH_ZIP = '../input/sr32power2mel384_111/train_img.zip' 

    data = np.load(PATH_ZIP)
    df = pd.read_csv('../input/train_tp.csv')
    # bin make
    df['dif_f'] = df.f_max - df.f_min
    df['dif_f'] = df['dif_f'].astype(int)
    df['bins'] = pd.cut(df.dif_f, 15, labels=False)
    df['file_name'] = 'file_name'
    
    for i in data.files[1:]:
        recording_id, species_id, idx = i.split('.')[0], i.split('.')[1], i.split('.')[2]
        if (df.loc[int(idx), 'recording_id'] == recording_id.split('/')[1] and df.loc[int(idx), 'species_id'] == int(species_id)):
            df.loc[int(idx),'file_name'] = i
        else:
            print(idx, recording_id, i)

    if args.PARAM == 260:
        print('check_correct_fold 260')
        for f, (tr, vl) in enumerate(skf.split(data.files[1:], df.bins.values)): # FOR 260 !!!!!!!!!!!!!
            print(len(tr), len(vl))
            tr_idx.append(tr)
            vl_idx.append(vl)
    else:
        for f, (tr, vl) in enumerate(skf.split(df, df.bins.values)): #  FOR 384 !!!!!!!!!!!!!!
            print(len(tr), len(vl))
            tr_idx.append(tr)
            vl_idx.append(vl)

    print('Correct !')
    # EBlite4, EBLite4_260, EBLite4_384
    # for i in range(5):    
    fold = 0
    # !!!!!!!!!!! IF 384 CHANGE TO (DF to --> DATA) PARAM
    if args.PARAM == 260:        
        model = EBLite4_260()
        showtime(model, fold, data, tr_idx, vl_idx, scaler, rand_data) 
    else:
        model = EBLite4_384()
        showtime(model, fold, df, tr_idx, vl_idx, scaler, rand_data)        