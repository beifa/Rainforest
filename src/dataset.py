import os
import glob
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from utils import mono_to_color 

PATH_ZIP = '../input/train_img.zip'

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

    def __init__(self, data, size, transform = False):        
        self.size = size     
        self.transform = transform
        self.zipdata = np.load(PATH_ZIP)
        self.data = data # list name npy,
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        name = self.data[index]      
     
        img = self.zipdata[name]  
        sci_id = name.split('.')[1]
        target = np.zeros(24)
        target[int(sci_id)] = 1
        if self.transform is not None:
            pass
        img = mono_to_color(img)       
        if self.size is not None:
            img = cv2.resize(img, (224, self.size)) 
            
        img = img / 255.0
        img = img.transpose(2, 0, 1).astype(np.float32)        
             
        return torch.tensor(img).float(),torch.tensor(target.astype(np.float16)).float() #check


# class RFDataset_test(Dataset):
#     """
#     .npy cut by 6sec one file have shape(10, *, *)    
#     """

#     def __init__(self, path, size, transform = None):
#         self.data = np.asarray(path)
#         self.size = size        
#         self.transform = transform
        
    
#     def __len__(self):
#         return self.data.shape[0]    
    
#     def __getitem__(self, index):
#         temp = []
#         name = self.data[index]   
#         img = np.load(name)          
#         name_img = os.path.basename(name)
        
#         if self.transform is not None:
#             pass

#         for i in range(len(img)):
#             image = mono_to_color(img[i]) 
#             if self.size is not None:
#                 image = cv2.resize(image, (576, self.size))
#             image = image / 255.0
#             image = image.transpose(2, 0, 1).astype(np.float32) 
#             temp.append(image)        
#         return torch.tensor(temp).float(), name_img.split('.')[0]


class RFDataset_test(Dataset):
    """
    .npy cut by 6sec one file have shape(10, *, *)    
    """

    def __init__(self, size, transform = None):
      self.p1 = np.load('../input/test_img_p1.zip')
      self.p2 = np.load('../input/test_img_p2.zip')
      
      self.size = size
      self.transform = transform
        
    
    def __len__(self):        
        return len(self.p1.files[1:]) + len(self.p2.files[1:])
    
    def __getitem__(self, index):
        temp = []
        
        if index <= 999:
          name = self.p1.files[1:][index]          
          name_file = os.path.basename(name)
          img = self.p1[name]
          # print(name, name_file)
        else:
          name = self.p2.files[1:][index-1000]
          name_file = os.path.basename(name)
          img = self.p2[name]                
        
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


    