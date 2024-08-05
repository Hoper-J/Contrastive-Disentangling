import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import normalize

from .resnet import get_resnet
    

class Network(nn.Module):
    def __init__(self, backbone, feature_num=128, hidden_dim=128):
        super(Network, self).__init__()
        self.resnet = get_resnet(backbone)
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.BatchNorm1d(self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, hidden_dim),
        )
        self.feature_predictor = nn.Sequential(
            nn.Linear(hidden_dim, self.resnet.rep_dim),
            nn.BatchNorm1d(self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, feature_num),
            nn.BatchNorm1d(feature_num),
            nn.Softmax(dim=1) 
        )
        

    def forward(self, x1, x2):
        h1 = self.resnet(x1)
        h2 = self.resnet(x2)

        z1 = self.instance_projector(h1)
        z2 = self.instance_projector(h2)
        
        f1 = self.feature_predictor(z1)
        f2 = self.feature_predictor(z2)
        
        return z1, z2, f1, f2
        

    def extract_backbone_and_feature(self, x):
        h = self.resnet(x)
        z = self.instance_projector(h)
        f = self.feature_predictor(z)
        return h, f