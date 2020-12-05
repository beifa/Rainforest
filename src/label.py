import os
import pandas as pd 
import numpy as np 

if __name__ == "__main__":
    # loads data
    tp = pd.read_csv(os.path.join('../input', 'train_tp.csv'))
    fp = pd.read_csv(os.path.join('../input', 'train_fp.csv'))

    # target 
    tp = tp[['recording_id','species_id']]
    fp = fp[['recording_id','species_id']]

    fp['species_id'] = -1

    label = pd.concat((tp, fp))

    for i in range(24):
        label['s'+str(i)] = 0
        label.loc[label.species_id==i,'s'+str(i)] = 1
        
    label.drop('species_id', axis = 1, inplace = True)
    label = label.reset_index(drop=True)
    label.to_csv('../input/label.csv', index =False)
    print('Correct !')