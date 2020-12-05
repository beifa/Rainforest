import torch
import torch.nn as nn
import pretrainedmodels

class Res50(nn.Module):
    
    def __init__(self):
        super(Res50, self).__init__()        
        self.model = pretrainedmodels.__dict__['resnet50'](pretrained = 'imagenet')
        # list(resnet50.children())[:-2]
        self.in_f = self.model.last_linear.in_features
        self.model.last_linear = nn.Identity()
        self.l1 = nn.Linear(self.in_f, 24)
        
        
    def forward(self, x):
        # for test add dim becous 20 bin
        if len(torch.squeeze(x).shape) == 4:
            x = torch.squeeze(x)
        else:
            x
        bs, _, _, _ = x.shape
        x = self.model(x)       
        x = x.view(bs, -1)
        x = self.l1(x)
        return x