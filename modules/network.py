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
        

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)
        
        out_i = self.class_projector(h_i)
        out_j = self.class_projector(h_j)

        if self.use_variational:
            c_i = out_i.predictive.probs 
            c_j = out_j.predictive.probs
        else:
            c_i = out_i
            c_j = out_j
        
        entropy_loss = 0.5 * (self.entropy_regularization(c_i) + self.entropy_regularization(c_j))
        
        return z_i, z_j, out_i, out_j, entropy_loss
    
    def entropy_regularization(self, probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return torch.mean(entropy)

    def forward_cluster(self, x):
        f = self.resnet(x)
        c = self.class_projector(f)
        if self.use_variational:
            c = c.logit_predictive.loc
        c = torch.argmax(c, dim=1)
        return c
