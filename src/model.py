import torch
import geffnet
import torch.nn as nn
import pretrainedmodels

sigmoid = torch.nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
    
swish_layer = Swish_module()



#default version
class EBlite4(nn.Module):
  """
  score all default:
                    260 best 0.878
                    360     0.862                    
  """
  def __init__(self):
    super().__init__()
    self.model = geffnet.create_model('tf_efficientnet_lite4', pretrained=True) 
    in_ch = self.model.classifier.in_features #1536
    self.myfc = nn.Sequential(nn.Linear(in_ch, 24))
    self.model.classifier = nn.Identity()   
      
  def extract(self, x):        
    x = self.model(x)        
    return x
      
  def forward(self, x):    
    if len(torch.squeeze(x).shape) == 4:
        x = torch.squeeze(x)
    else:
        x
    bs, _, _, _ = x.shape    
    x = self.extract(x)
    x = self.myfc(x)
    return x  

# modify, add layes

class EBLite4_260(nn.Module):

  def __init__(self):
    super().__init__()
    self.model = geffnet.create_model('tf_efficientnet_lite4', pretrained=True) 
    self.model.global_pool = nn.AdaptiveAvgPool2d((1,1)) 
    in_ch = self.model.classifier.in_features
    self.myfc = nn.Sequential(
        nn.Dropout(0.17418),
        nn.Linear(in_ch, 785),
        nn.BatchNorm1d(785),
        Swish_module(),
        nn.Dropout(0.12),         
        nn.Linear(785, 1038),
        nn.BatchNorm1d(1038),
        Swish_module(),
        nn.Dropout(0.3763722),       
        nn.Linear(1038, 24)
    )   
    
    self.model.classifier = nn.Identity()   
      
  def extract(self, x):        
    x = self.model(x)        
    return x
      
  def forward(self, x):    
    if len(torch.squeeze(x).shape) == 4:
        x = torch.squeeze(x)
    else:
        x
    bs, _, _, _ = x.shape    
    x = self.extract(x)
    x = self.myfc(x)
    return x 

class EBLite4_384(nn.Module):

  def __init__(self):
    super().__init__()
    self.model = geffnet.create_model('tf_efficientnet_lite4', pretrained=True) 
    self.model.global_pool = nn.AdaptiveAvgPool2d((1,1)) 
    in_ch = self.model.classifier.in_features
    self.myfc = nn.Sequential(
        nn.Dropout(0.29),
        nn.Linear(in_ch, 1307),
        nn.BatchNorm1d(1307),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1307, 1307),
        nn.BatchNorm1d(1307),
        nn.ReLU(),
        nn.Dropout(0.2),         
        nn.Linear(1307, 24)
    )   
    
    self.model.classifier = nn.Identity()   
      
  def extract(self, x):        
    x = self.model(x)        
    return x
      
  def forward(self, x):    
    if len(torch.squeeze(x).shape) == 4:
        x = torch.squeeze(x)
    else:
        x
    bs, _, _, _ = x.shape    
    x = self.extract(x)  
    x = self.myfc(x)
    return x
