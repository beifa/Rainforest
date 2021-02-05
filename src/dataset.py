import os
import glob
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from utils import mono_to_color, rand_window 

# class RFDataset(Dataset):

#     def __init__(self, path, data, size, transform = False):
#         self.data = data
#         self.size = size
#         self.path = path
#         self.transform = transform
#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, index):
#         name = self.data.recording_id[index]
#         img = np.load(os.path.join(self.path, name) + '.npy')
#         if self.transform is not None:
#             pass
#         img = mono_to_color(img)       
#         if self.size is not None:
#             img = cv2.resize(img, (576, self.size))
            
#         img = img / 255.0
#         img = img.transpose(2, 0, 1).astype(np.float32)        
#         target = self.data.loc[index].values[1:]        
#         return torch.tensor(img).float(),torch.tensor(target.astype(np.float16)).float() #check


# check work with zip
class RFDataset(Dataset):

    def __init__(self, data, path, version = 'v1', size = None, rand = None, transform = None):        
        self.size = size     
        self.transform = transform
        self.rand = rand
        self.version = version
        self.zipdata = np.load(path)
        self.data = data # list name npy,
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        name = self.data[index]        
        if self.rand is not None:           
            img_ori = self.zipdata[name]
            img = rand_window(img_ori, self.version) 
        else:
            img = self.zipdata[name] 

        sci_id = name.split('.')[1]
        target = np.zeros(24)
        target[int(sci_id)] = 1

        img = mono_to_color(img)       
        if self.size is not None:            
            img = cv2.resize(img, (224, self.size)) 

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
            
        img = img / 255.0
        img = img.transpose(2, 0, 1).astype(np.float32)        
             
        return torch.tensor(img).float(),torch.tensor(target.astype(np.float16)).float() 


class RFDataset_test(Dataset):
    """
    .npy cut by 6sec one file have shape(10, *, *)    
    """

    def __init__(self, name, size, transform = None):
        p1 = glob.glob(f'../input/{name}/test_img_p1/test/*.npy')
        p2 = glob.glob(f'../input/{name}/test_img_p2/test/*.npy')
        p1.extend(p2)
        self.p1 = p1
        
        self.size = size
        self.transform = transform
        
    
    def __len__(self):        
        return len(self.p1) 
    
    def __getitem__(self, index):
        temp = []        
       
        name = self.p1[index]          
        name_file = os.path.basename(name)
        img = np.load(name)
  
        if self.transform is not None:
            pass

        for i in range(len(img)):
            image = mono_to_color(img[i]) 
            if self.size is not None:
                image = cv2.resize(image, (224, self.size))
            image = image / 255.0
            image = image.transpose(2, 0, 1).astype(np.float32) 
            temp.append(image)        
        return torch.tensor(temp).float(), name_file.split('.')[0]