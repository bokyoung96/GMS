"""
TEAM 고객의 미래를 새롭게

코드 목적: cs 데이터를 AutoEncoder로 분석합니다.
참조: https://github.com/techshot25/Autoencoders
"""
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


class AutoEncoder(nn.Module):
    def __init__(self, in_shape, enc_shape, dropout):
        super(AutoEncoder, self).__init__()

        self.dropout = dropout

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 270),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(270, 135),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(135, 67),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(67, 32),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(8, enc_shape)
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 8),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(32, 67),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(67, 135),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(135, 270),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(270, in_shape)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class AELoader:
    def __init__(self, item_type: str,
                 test_size: float = 0.2,
                 enc_shape: int = 3,
                 dropout: float = 0.2):
        self.item_type = item_type
        self.test_size = test_size
        self.enc_shape = enc_shape
        self.dropout = dropout
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print("***** USED DEVICE WILL BE: {} *****".format(self.device))

    @property
    def data_loader(self) -> pd.DataFrame:
        with open('./res_pp_categories/res_pp_{}.pkl'.format(self.item_type), 'rb') as f:
            data = pickle.load(f)
        print("\n ***** DATA LOADED ***** \n")
        return data

    def to_torch(self):
        data = self.data_loader.values
        return torch.from_numpy(data).to(self.device)

    def data_split(self):
        # NOTE: NOT USED CURRENTLY. DEVELOPMENT FOR EARLY STOPPING & VALIDATION CHECK REQUIRED.
        data = TensorDataset(self.to_torch())
        x_train, x_valid = train_test_split(
            data, test_size=self.test_size, random_state=42)
        loader_train = DataLoader(x_train, batch_size=64, shuffle=True)
        loader_valid = DataLoader(x_valid, batch_size=64, shuffle=True)
        return loader_train, loader_valid

    @staticmethod
    def ae_train(model, error, optimizer, n_epochs, x):
        model.train()
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            output = model(x)
            loss = error(output, x)
            loss.backward()
            optimizer.step()

            if epoch % int(0.01*n_epochs) == 0:
                print(f'Epoch {epoch} \t Loss: {loss.item():.4g}')

            if loss.item() < 0.05:
                print("\n ***** LOSS UNDER 0.05 - PROCESS FINISHED ***** \n")
                break

    def ae_run(self, n_epochs: int = 100):
        x = self.to_torch()
        auto_encoder = AutoEncoder(in_shape=x.shape[1],
                                   enc_shape=self.enc_shape,
                                   dropout=self.dropout).double().to(self.device)
        error = nn.MSELoss()
        optimizer = optim.Adam(auto_encoder.parameters())

        self.ae_train(auto_encoder, error, optimizer, n_epochs, x)
        with torch.no_grad():
            encoded = auto_encoder.encode(x)
            decoded = auto_encoder.decode(encoded)
            mse = error(decoded, x).item()
            enc = encoded.cpu().detach().numpy()
            dec = decoded.cpu().detach().numpy()
        return mse, enc, dec

    def ae_plot(self, enc: np.ndarray):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.scatter(enc[:, 0], enc[:, 1], enc[:, 2], cmap=plt.cm.jet)

        plt.title("ENCODED DATA: {}".format(self.item_type))
        plt.show()


class KMLoader(AELoader):
    def __init__(self,
                 item_type: str = 'ast',
                 test_size: float = 0.2,
                 enc_shape: int = 3,
                 dropout: float = 0.2,
                 n_epochs: int = 100):
        super().__init__(item_type, test_size, enc_shape, dropout)
        self.n_epochs = n_epochs
        self.enc = pd.DataFrame(self.ae_run(n_epochs=self.n_epochs)[1])

    def km_n_clusters(self):
        # NOTE: HIGH COST
        for n_clusters in range(2, 6):
            model = KMeans(n_clusters=n_clusters,
                           random_state=42).fit(self.enc)
            score = silhouette_score(self.enc, model.labels_)
            print("***** Cluster: {} || Silhouette index: {}".format(n_clusters, score))
        return

    def km_run(self):
        df = self.enc.copy()
        model = KMeans(n_clusters=5, random_state=42).fit(self.enc)
        df['LABEL'] = model.labels_

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0],
                   df.iloc[:, 1],
                   df.iloc[:, 2],
                   c=df.LABEL,
                   s=10,
                   cmap='rainbow',
                   alpha=1)
        plt.show()
        return df


# if __name__ == "__main__":
#     loader = AELoader(item_type='ast')
#     mse, enc, dec = loader.ae_run(n_epochs=100)
#     loader.ae_plot(enc=enc)
