import math
import torch
import torch.nn as nn
from torch.nn.functional import normalize


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        """
        Initialize the InstanceLoss module.

        Parameters:
        - batch_size: Size of the batch.
        - temperature: Temperature parameter for scaling similarities.
        - device: The device on which the computations are performed.
        """
        super(InstanceLoss, self).__init__()
        
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        
        self.N = 2 * batch_size
        self.mask = self._mask_correlated_samples()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _mask_correlated_samples(self):
        """
        Create a mask to exclude correlated samples from the loss calculation.

        Returns:
        - mask: Boolean mask excluding correlated samples.
        """
        mask = torch.eye(self.N, dtype=torch.bool).to(self.device)
        mask[:self.batch_size, self.batch_size:] = torch.eye(self.batch_size, dtype=torch.bool).to(self.device)
        mask[self.batch_size:, :self.batch_size] = torch.eye(self.batch_size, dtype=torch.bool).to(self.device)
        mask = ~mask
        return mask

    def forward(self, z1, z2):
        """
        Calculate the instance loss.

        Parameters:
        - z1: Latent representations of the first view.
        - z2: Latent representations of the second view.

        Returns:
        - loss: Calculated instance loss.
        """
        z1 = normalize(z1, dim=1)
        z2 = normalize(z2, dim=1)
        
        z = torch.cat((z1, z2), dim=0)
    
        sim = torch.matmul(z, z.T) / self.temperature
        positive_samples = torch.cat((torch.diag(sim, self.batch_size), torch.diag(sim, -self.batch_size)), dim=0).reshape(self.N, 1)
        negative_samples = sim[self.mask].reshape(self.N, -1)
        
        labels = torch.zeros(self.N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        
        return loss


class FeatureLoss(nn.Module):
    def __init__(self, feature_num, temperature, device):
        """
        Initialize the FeatureLoss module.

        Parameters:
        - feature_num: Number of features.
        - temperature: Temperature parameter for scaling similarities.
        - device: The device on which the computations are performed.
        """
        super(FeatureLoss, self).__init__()
        
        self.feature_num = feature_num
        self.temperature = temperature
        self.device = device
        
        self.K = 2 * feature_num
        self.mask = self._mask_correlated_features()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _mask_correlated_features(self):
        """
        Create a mask to exclude correlated features from the loss calculation.

        Returns:
        - mask: Boolean mask excluding correlated features.
        """
        mask = torch.eye(self.K, dtype=torch.bool).to(self.device)
        mask[:self.feature_num, self.feature_num:] = torch.eye(self.feature_num, dtype=torch.bool).to(self.device)
        mask[self.feature_num:, :self.feature_num] = torch.eye(self.feature_num, dtype=torch.bool).to(self.device)
        mask = ~mask
        return mask

    def _normalized_entropy_loss(self, predictions, epsilon=1e-12):
        """
        Calculate the normalized entropy loss for feature predictions.

        Parameters:
        - predictions: Predicted feature probabilities from the feature predictor (after sigmoid activation).
        - epsilon: Small value to avoid division by zero.

        Returns:
        - normalized_entropy: Mean normalized entropy loss.
        """
        entropy = -predictions * torch.log(predictions + epsilon) - (1 - predictions) * torch.log(1 - predictions + epsilon)    
        max_entropy = torch.log(torch.tensor(2.0))
        normalized_entropy = entropy / max_entropy
        return torch.mean(normalized_entropy)

    def forward(self, f1, f2):
        """
        Calculate the feature loss.

        Parameters:
        - f1: Feature predictions from the first view (output of the sigmoid-activated feature predictor).
        - f2: Feature predictions from the second view (output of the sigmoid-activated feature predictor).

        Returns:
        - loss: Calculated feature loss including normalized entropy loss for diversity.
        """
        ne1 = self._normalized_entropy_loss(f1)
        ne2 = self._normalized_entropy_loss(f2)
        neloss = (ne1 + ne2) / 2
        
        f1 = f1.T
        f2 = f2.T
        f = torch.cat((f1, f2), dim=0)

        sim = self.similarity_f(f.unsqueeze(1), f.unsqueeze(0)) / self.temperature
        positive_features = torch.cat((torch.diag(sim, self.feature_num), torch.diag(sim, -self.feature_num)), dim=0).reshape(self.K, 1)
        negative_features = sim[self.mask].reshape(self.K, -1)

        labels = torch.zeros(self.K).to(positive_features.device).long()
        logits = torch.cat((positive_features, negative_features), dim=1)
        loss = self.criterion(logits, labels)
        
        return loss - neloss
