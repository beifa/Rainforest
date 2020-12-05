import os
import glob
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from utils import mono_to_color 


class RFDataset(Dataset):

    def __init__(self, path, data, size, transform = False):
        self.data = data
        self.size = size
        self.path = path
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        name = self.data.recording_id[index]
        img = np.load(os.path.join(self.path, name) + '.npy')
        if self.transform is not None:
            pass
        img = mono_to_color(img)       
        if self.size is not None:
            img = cv2.resize(img, (550, self.size))
            
        img = img / 255.0
        img = img.transpose(2, 0, 1).astype(np.float32)        
        target = self.data.loc[index].values[1:]        
        return torch.tensor(img).float(),torch.tensor(target.astype(np.float16)).float() #check


class RFDataset_test(Dataset):
    """
    .npy cut by 6sec one file have shape(10, *, *)    
    """

    def __init__(self, size, transform = None):
        self.data = np.asarray(glob.glob( '../input/test_npy/*.npy'))
        self.size = size
        self.transform = transform
        
    
    def __len__(self):
        return self.data.shape[0]    
    
    def __getitem__(self, index):
        temp = []
        name = self.data[index]   
        img = np.load(name)          
        name_img = os.path.basename(name)
        
        if self.transform is not None:
            pass

        for i in range(len(img)):
            image = mono_to_color(img[i]) 
            if self.size is not None:
                image = cv2.resize(image, (550, self.size))
            image = image / 255.0
            image = image.transpose(2, 0, 1).astype(np.float32) 
            temp.append(image)        
        return torch.tensor(temp).float(), name_img.split('.')[0]