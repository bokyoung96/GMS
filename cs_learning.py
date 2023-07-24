import torch
import torch.nn as nn

from sklearn.cluster import KMeans


class CustomerLearning:
    pass


class AutoEncoder(nn.Module, CustomerLearning):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class KMeansClustering(CustomerLearning):
    def __init__(self):
        super().__init__()

    def k_means_clustering(self):
        return
