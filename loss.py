import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.functional import normalize

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        # Create a full matrix with False
        mask = torch.eye(N, dtype=torch.bool).to(self.device)
    
        # Set the correlated samples to True
        mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool).to(self.device)
    
        # Invert mask for loss calculation (1s for negative samples)
        mask = ~mask
        return mask


    def forward(self, z1, z2):
        z1 = normalize(z1, dim=1)
        z2 = normalize(z2, dim=1)
        
        N = 2 * self.batch_size
        z = torch.cat((z1, z2), dim=0)
    
        sim = torch.matmul(z, z.T) / self.temperature
        sim_1_2 = torch.diag(sim, self.batch_size)
        sim_2_1 = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_1_2, sim_2_1), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    
class VarientialFeatureLoss(nn.Module):
    def __init__(self, feature_num, temperature, device, use_variational=True, var_weight=0.5):
        super(VarientialFeatureLoss, self).__init__()
        self.feature_num = feature_num
        self.temperature = temperature
        self.device = device
        self.use_variational = use_variational
        self.var_weight = var_weight

        self.mask = self.mask_correlated_features(feature_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
    def mask_correlated_features(self, feature_num):
        K = 2 * feature_num
        # Create a full matrix with False
        mask = torch.eye(K, dtype=torch.bool).to(self.device)
    
        # Set the correlated features to True
        mask[:feature_num, feature_num:] = torch.eye(feature_num, dtype=torch.bool).to(self.device)
        mask[feature_num:, :feature_num] = torch.eye(feature_num, dtype=torch.bool).to(self.device)
    
        # Invert mask for loss calculation (1s for negative features)
        mask = ~mask
        return mask

    def forward(self, f1, f2):
        variational_loss = torch.tensor(0.0, device=self.device)
        if self.use_variational:
            feature_pred1 = torch.argmax(f1.logit_predictive.loc, dim=1)
            feature_pred2 = torch.argmax(f2.logit_predictive.loc, dim=1)
            variational_loss1 = f1.train_loss_fn(feature_pred2.detach())
            variational_loss2 = f2.train_loss_fn(feature_pred1.detach())
            variational_loss = self.var_weight * (variational_loss1 + variational_loss2) / 2

            f1 = f1.predictive.probs 
            f2 = f2.predictive.probs

        else:
            f1 = f1
            f2 = f2
        
        p1 = f1.sum(0).view(-1)
        p1 /= p1.sum()
        ne1 = math.log(p1.size(0)) + (p1 * torch.log(p1)).sum()
        p2 = f2.sum(0).view(-1)
        p2 /= p2.sum()
        ne2 = math.log(p2.size(0)) + (p2 * torch.log(p2)).sum()
        ne_loss = ne1 + ne2
        
        f1 = f1.t()
        f2 = f2.t()
        K = self.feature_num * 2
        f = torch.cat((f1, f2), dim=0)

        sim = self.similarity_f(f.unsqueeze(1), f.unsqueeze(0)) / self.temperature
        sim_1_2 = torch.diag(sim, self.feature_num)
        sim_2_1 = torch.diag(sim, -self.feature_num)

        positive_clusters = torch.cat((sim_1_2, sim_2_1), dim=0).reshape(K, 1)
        negative_clusters = sim[self.mask].reshape(K, -1)

        labels = torch.zeros(K).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= K

        return loss + ne_loss, variational_loss