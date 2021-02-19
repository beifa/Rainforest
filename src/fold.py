"""
Idea folds:

    1. we train only tp and use(Group or Str.Fols)
    2. Uniques: find count un uiques mark and group by them-->BAD
    3. Group by recording_id and target , unbalance-->BAD
    4. Diff f_max&min, try 32 variant cool!
    4. by target
    
    There is a problem:
         when forming folds for each s0-s23 feature, the target value is approximately 50, this is 0.00555
         and it turns out that when folded, the target simply does not fall into the fold.
         for NN this probably won't be a problem
    
    There are many ideas, but you need to stop and try while these six options look the most attractive

TODO:
   Test:
      Cool idea make stratification by uniques(in group array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
      will be in folds) PS and 'Group by recording id and target' we check by strat.


"""

import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import GroupKFold, StratifiedKFold
PATH_CSV = '../input/'

# fold
"""
fold  species_id
0     21            350
1     13            350
      15            350
2     0              50
3     1             348
      3             350
      4             350
      8             350
      16            350
      18            350
6     0             300
      12            350
      14            350
      20            340
      22            330
8     9             334
      23            700
12    2             337
      5             350
      6             350
      7             350
      11            350
      17            668
      19            342
13    10            348
Name: fold, dtype: int64

"""

key_fold = {    
    750.0000 : 1,
    843.7500 : 0,
    1031.2500: 1,
    1312.5000: 1,
    1500.0000: 3,
    1593.7500: 5,
    1781.2500: 3,
    1875.0000: 3,
    2343.7500: 2,
    2531.2500: 7,
    2625.0000: 7,
    2718.7500: 5,
    3000.0000: 5,
    3100.7799999999997: 7,
    3375.0000: 5,
    3843.7500: 7,
    3875.9764999999998: 7,
    3875.9800000000005: 7,
    4048.2400: 6,
    4048.2421999999997: 6,
    4125.0000: 6,
    4781.2500: 5,
    4875.0000: 6,
    5167.94: 6,
    5167.968800000001 : 6,
    5624.950000000001 : 7,    
    5625.0000: 7,
    6093.7500: 7,
    6187.5000: 7,
    6750.0000: 7,
    9905.239000000001: 8,
    9905.2735: 8
}


if __name__ == "__main__":
    
    tp = pd.read_csv(os.path.join(PATH_CSV, 'train_tp.csv'))
    fp = pd.read_csv(os.path.join(PATH_CSV, 'train_fp.csv'))
    tp['mark'] = 1
    fp['mark'] = -1
    df = pd.concat([tp, fp])
    df = df.sample(frac = 1).reset_index(drop=True)

    # find diff time
    df['diff_time'] = df.f_max - df.f_min
    df['groups'] = df['diff_time'].map(key_fold)

    # v1 GroupFold, 3-6 count target but 350 val
    # each fold train and valid for small group and predict
    # all fold have all mark target 
    # pred for current group mark each fold
    # this no crosval idea but try it, check not make mean / 5 submission
    gkf = GroupKFold(n_splits = 5)
    df['gfold'] = -1
    for fold, (t_idx, v_idx) in enumerate(gkf.split(X=df, y= df.species_id.values, groups = df.groups.values)):
        print(len(t_idx), len(v_idx))
        df.loc[v_idx, 'gfold'] = fold

    # v2 StrFold fold = 4 give ~ 80 each values valid, fold= 5 ~ 50-60, fold= 3 ~ 100
    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=13)
    df['sfold'] = -1

    for fold, (t_idx, v_idx) in enumerate(skf.split(X=df, y= df.groups.values)):
        print(len(t_idx), len(v_idx))
        df.loc[v_idx, 'sfold'] = fold 
    
    groups = df['recording_id'].values
    df['gkffold'] = -1
    for f, (tr,vl) in enumerate(gkf.split(df, df.species_id, groups)):
        print(len(t_idx), len(v_idx))
        df.loc[vl, 'gkffold'] = f 

    df.to_csv(os.path.join(PATH_CSV, 'gfold_sfold_gkffold_df.csv'), index = False)
    
    # v3 only tp Str.Fould by target
    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=13)
    tp = pd.read_csv(os.path.join(PATH_CSV, 'train_tp.csv'))
    tp = tp.sample(frac = 1).reset_index(drop=True)
    tp['fold'] = -1


    for fold, (t_idx, v_idx) in enumerate(skf.split(X=tp, y= tp.species_id.values)):
        print(len(t_idx), len(v_idx))
        tp.loc[v_idx, 'fold'] = fold  
    tp.to_csv(os.path.join(PATH_CSV, 'tp_sfold.csv'), index = False)
    
    #--------------------
    # groups = tp['recording_id'].values
    # tp['gfold'] = -1
    # for f, (tr,vl) in enumerate(gkf.split(tp, tp.species_id, groups)):
    #     tp.loc[vl, 'gfold'] = f
    # tp.to_csv(os.path.join(PATH_CSV, 'tp_gfold.csv'), index = False)

    # label = tp[['recording_id','species_id']].copy()   

    # for i in range(24):
    #     label['s'+str(i)] = 0
    #     label.loc[label.species_id==i,'s'+str(i)] = 1
        
    # label.drop('species_id', axis = 1, inplace = True)
    # label.to_csv(os.path.join(PATH_CSV, 'label_tp.csv'), index = False)

    tp = pd.read_csv(os.path.join(PATH_CSV, 'train_tp.csv'))
    fp = pd.read_csv(os.path.join(PATH_CSV, 'train_fp.csv'))
    fp['species_id'] = -1 
    df = pd.concat([tp, fp])
    df = df.sample(frac = 1).reset_index(drop=True)

    df = df.set_index('recording_id').loc[tp.recording_id]
    df = df.reset_index(drop = False)

    label = df[['recording_id','species_id']].copy()    
    for i in range(24):
        label['s'+str(i)] = 0
        label.loc[label.species_id==i,'s'+str(i)] = 1

    groups = df['recording_id'].values
    df['gfold'] = -1
    for f, (tr,vl) in enumerate(gkf.split(df, df.species_id, groups)):
        df.loc[vl, 'gfold'] = f
    df.to_csv(os.path.join(PATH_CSV, 'tp_gfold.csv'), index = False)
        
    label.drop('species_id', axis = 1, inplace = True)
    label.to_csv(os.path.join(PATH_CSV, 'label_tp.csv'), index = False)
    print(df.shape, label.shape)