import torch
import torch.nn as nn

from .resnet import get_resnet


class Network(nn.Module):
    def __init__(self, backbone, feature_num=128, hidden_dim=128):
        """
        Initialize the Network architecture.

        Parameters:
        - backbone: The backbone network to extract features (e.g., ResNet).
        - feature_num: Number of features predicted by the feature predictor.
        - hidden_dim: Dimensionality of the hidden layers.
        """
        super(Network, self).__init__()
        
        self.resnet = get_resnet(backbone)
        
        # Instance projector to process the backbone's output
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim, bias=False),
            nn.BatchNorm1d(self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, hidden_dim)
        )
        
        # Feature predictor to output feature predictions (using Sigmoid activation)
        self.feature_predictor = nn.Sequential(
            nn.Linear(hidden_dim, self.resnet.rep_dim, bias=False),
            nn.BatchNorm1d(self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, feature_num, bias=False),
            nn.BatchNorm1d(feature_num),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """
        Forward pass for two input views.

        Parameters:
        - x1: First view input tensor.
        - x2: Second view input tensor.

        Returns:
        - z1, z2: Latent representations for both views.
        - f1, f2: Feature predictions for both views.
        """
        h1 = self.resnet(x1)
        h2 = self.resnet(x2)

        z1 = self.instance_projector(h1)
        z2 = self.instance_projector(h2)
        
        f1 = self.feature_predictor(z1)
        f2 = self.feature_predictor(z2)
        
        return z1, z2, f1, f2

    def extract_backbone_and_feature(self, x):
        """
        Extract backbone features and feature predictions for a single input.

        Parameters:
        - x: Input tensor.

        Returns:
        - h: Backbone feature output.
        - f: Feature predictions.
        """
        h = self.resnet(x)
        z = self.instance_projector(h)
        f = self.feature_predictor(z)
        
        return h, f
