import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset


from model import Res50
from dataset import RFDataset_test


LABEL = '../input/label.csv'
PATH_MODEL = '../models'
PATH_SUBMIT = '../submission'
PATH_CSV = '../input'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, required=True, help = 'name_model_size_epoch_typefold_scheduler_mixadd')
    parser.add_argument('--fold_type', type=str, required=True, help = 'version fold')
    parser.add_argument('--model_type', type=str, required=True, help = 'Model type [best, final]')
    args, _ = parser.parse_known_args()
    return args


def main():

    #v1
    sub = pd.read_csv(os.path.join(PATH_CSV, 'sample_submission.csv'))
    sub = sub.set_index('recording_id')
    sub *= 0
    
    #v2
    sub2 = pd.read_csv(os.path.join(PATH_CSV, 'sample_submission.csv'))
    sub2 = sub2.set_index('recording_id')
    sub2 *= 0

    test_dataset = RFDataset_test(size = 128)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)

    # loads model
    for f in range(1):
        print('Fold: ', f + 1)

        if args.model_type == 'best':
            model_type = os.path.join(PATH_MODEL, f'{args.kernel}_best_fold_{f}.pth')
        if args.model_type == 'final':
            model_type = os.path.join(PATH_MODEL, f'{args.kernel}_final_fold_{f}.pth')
        
        model = MODEL()
        print(f'Loads model: {model_type}')
        model = model.to(device)
        model.load_state_dict(torch.load(model_type), strict=True)
        model.eval()

        # models.append(model)
        with torch.no_grad():
            bar = tqdm(test_loader)
            for (img, name) in bar:
                img = img.to(device)
                y_ = model(img)
                #
                prob2 = torch.max(nn.Sigmoid()(y_), dim=0)[0]
                pred2  =prob2.detach().cpu().numpy()
                #
                prob = torch.max(y_, dim=0)[0]
                pred  =prob.detach().cpu().numpy() 
                
                sub.loc[name[0]] += pred 
                sub2.loc[name[0]] += pred2 
    # save
    if args.fold_type == 'gfold':        
        sub = sub.reset_index()        
        sub2 = sub2.reset_index()
    else:
        sub.iloc[:, 1:] /= 5
        sub = sub.reset_index()
        sub2.iloc[:, 1:] /= 5
        sub2 = sub2.reset_index()

    sub.to_csv(os.path.join(PATH_SUBMIT, f'sub_{args.kernel}_{args.model_type}_{args.fold_type}.csv'), index=False)
    sub2.to_csv(os.path.join(PATH_SUBMIT, f'sub2_{args.kernel}_{args.model_type}_{args.fold_type}.csv'), index=False)    



if __name__ == "__main__":
    args = parse_args()
    MODEL = Res50
    device = torch.device('cuda')
    main()