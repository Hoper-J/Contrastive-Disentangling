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
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

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
        
        return loss


class FeatureLoss(nn.Module):
    def __init__(self, feature_num, temperature, device):
        super(FeatureLoss, self).__init__()
        self.feature_num = feature_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_features(feature_num)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
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

    def normalized_entropy_loss(self, predictions, epsilon=1e-12):
        entropy = -predictions * torch.log(predictions + epsilon) - (1 - predictions) * torch.log(1 - predictions + epsilon)    
        max_entropy = torch.log(torch.tensor(2.0))
        normalized_entropy = entropy / max_entropy
    
        return torch.mean(normalized_entropy)

    def forward(self, f1, f2):
        ne1 = self.normalized_entropy_loss2(f1)
        ne2 = self.normalized_entropy_loss2(f2)
        neloss = (ne1 + ne2) / 2
        
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
        
        return loss - neloss