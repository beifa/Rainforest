import torch
import geffnet
import torch.nn as nn
import pretrainedmodels

class Res50(nn.Module):
    
    def __init__(self):
        super(Res50, self).__init__()        
        self.model = pretrainedmodels.__dict__['resnet50'](pretrained = 'imagenet')        
        in_ch = self.model.last_linear.out_features  
        self.myfc = nn.Linear(in_ch, 24)
        self.model.fc = nn.Identity()   
       
    def extract(self, x):
        x = self.model(x)
        return x
        
    def forward(self, x):
        # for test add dim because 10 bin
        if len(torch.squeeze(x).shape) == 4:
            x = torch.squeeze(x)
        else:
            x
        bs, _, _, _ = x.shape
        x = self.extract(x)     
        x = x.view(bs, -1)
        x = self.myfc(x)
        return x

class EffB3(nn.Module):

  def __init__(self):
    super(EffB3, self).__init__()
    self.model = geffnet.create_model('efficientnet_b3', pretrained=True) 
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
    # x = x.view(bs, -1)   
    x = self.myfc(x)
    return x  