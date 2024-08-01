import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import normalize

from .resnet import get_resnet
from vbll.layers.classification import DiscClassification
    

class Network(nn.Module):
    def __init__(self, backbone, feature_num, batch_size, output_dim=128, use_variational=True):
        super(Network, self).__init__()
        self.resnet = get_resnet(backbone)
        self.use_variational = use_variational
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, output_dim),
        )
        if self.use_variational:
            self.feature_predictor = nn.Sequential(
                nn.Linear(output_dim, self.resnet.rep_dim),
                nn.ReLU(),
                DiscClassification(self.resnet.rep_dim, feature_num, 1. / (2 * batch_size), parameterization='diagonal', dof=1.)
            )
        else:
            self.feature_predictor = nn.Sequential(
                nn.Linear(output_dim, self.resnet.rep_dim),
                nn.ReLU(),
                nn.Linear(self.resnet.rep_dim, feature_num),
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
        z = normalize(self.instance_projector(h), dim=1)
        f = self.feature_predictor(z)
        if self.use_variational:
            f = f.logit_predictive.loc
        return h, f
