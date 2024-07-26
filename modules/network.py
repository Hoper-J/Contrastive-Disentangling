import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import normalize

from .resnet import get_resnet
from vbll.layers.classification import DiscClassification
    

class Network(nn.Module):
    def __init__(self, backbone, class_num, batch_size, output_dim=128, use_variational=True):
        super(Network, self).__init__()
        self.resnet = get_resnet(backbone)
        self.use_variational = use_variational
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, output_dim),
        )
        if self.use_variational:
            self.class_projector = nn.Sequential(
                nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
                nn.ReLU(),
                DiscClassification(self.resnet.rep_dim, class_num, 1. / (2 * batch_size), parameterization='diagonal', dof=1.)
            )
        else:
            self.class_projector = nn.Sequential(
                nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
                nn.ReLU(),
                nn.Linear(self.resnet.rep_dim, class_num),
                nn.Softmax(dim=1) 
            )
        

    def forward(self, x1, x2):
        h1 = self.resnet(x1)
        h2 = self.resnet(x2)

        z1 = normalize(self.instance_projector(h1), dim=1)
        z2 = normalize(self.instance_projector(h2), dim=1)
        
        v1 = self.class_projector(h1)
        v2 = self.class_projector(h2)
        
        return z1, z2, v1, v2
        

    def forward_cluster(self, x):
        f = self.resnet(x)
        c = self.class_projector(f)
        if self.use_variational:
            c = c.logit_predictive.loc
        c = torch.argmax(c, dim=1)
        return c
