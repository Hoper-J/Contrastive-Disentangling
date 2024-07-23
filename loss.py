import numpy as np
import torch
import torch.nn as nn
import math

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
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z1, z2):
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
    
class VarientialClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device, use_variational=True, var_weight=0.5):
        super(VarientialClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.use_variational = use_variational
        self.var_weight = var_weight

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
    def mask_correlated_clusters(self, class_num):
        K = 2 * class_num
        mask = torch.ones((K, K), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(class_num):
            mask[i, class_num + i] = False
            mask[class_num + i, i] = False
        return mask

    def forward(self, v1, v2):
        variational_loss = torch.tensor(0.0, device=self.device)
        if self.use_variational:
            class_pred1 = torch.argmax(v1.logit_predictive.loc, dim=1)
            class_pred2 = torch.argmax(v2.logit_predictive.loc, dim=1)
            variational_loss1 = v1.train_loss_fn(class_pred2.detach())
            variational_loss2 = v2.train_loss_fn(class_pred1.detach())
            variational_loss = self.var_weight * (variational_loss1 + variational_loss2) / 2

            c1 = v1.predictive.probs 
            c2 = v2.predictive.probs

        else:
            c1 = v1
            c2 = v2
        
        p1 = c1.sum(0).view(-1)
        p1 /= p1.sum()
        ne1 = math.log(p1.size(0)) + (p1 * torch.log(p1)).sum()
        p2 = c2.sum(0).view(-1)
        p2 /= p2.sum()
        ne2 = math.log(p2.size(0)) + (p2 * torch.log(p2)).sum()
        ne_loss = ne1 + ne2
        
        c1 = c1.t()
        c2 = c2.t()
        K = self.class_num * 2
        c = torch.cat((c1, c2), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_1_2 = torch.diag(sim, self.class_num)
        sim_2_1 = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_1_2, sim_2_1), dim=0).reshape(K, 1)
        negative_clusters = sim[self.mask].reshape(K, -1)

        labels = torch.zeros(K).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= K

        return loss + ne_loss, variational_loss