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

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
    
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
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
        N = 2 * class_num
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(class_num):
            mask[i, class_num + i] = False
            mask[class_num + i, i] = False
        return mask

    def forward(self, out_i, out_j):
        
        
        variational_loss = torch.tensor(0.0, device=self.device)
        if self.use_variational:
            c_i = out_i.predictive.probs 
            c_j = out_j.predictive.probs
        
            variational_loss_i = out_i.train_loss_fn(torch.argmax(out_j.logit_predictive.loc.detach(), dim=1))
            variational_loss_j = out_j.train_loss_fn(torch.argmax(out_i.logit_predictive.loc.detach(), dim=1))
            variational_loss = self.var_weight * (variational_loss_i + variational_loss_j) / 2

        else:
            c_i = out_i
            c_j = out_j
        
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j
        
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss, variational_loss