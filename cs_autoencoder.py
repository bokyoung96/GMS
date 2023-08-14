"""
TEAM 고객의 미래를 새롭게

코드 목적: cs 데이터를 AutoEncoder로 분석합니다.
참조: https://github.com/techshot25/Autoencoders
"""
import torch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from torch import nn, optim

from cs_preprocess import *

device = ('cuda' if torch.cuda.is_available() else 'cpu')

pp = CustomerPreprocess(item_type='ast')
data = pp.pp_load
x = torch.from_numpy(data).to(device)


class AutoEncoder(nn.Module):
    def __init__(self, in_shape, enc_shape):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


encoder = AutoEncoder(in_shape=372, enc_shape=3).double().to(device)
error = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters())


def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()

        if epoch % int(0.01*n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')

        if loss.item() < 0.05:
            print("\n ***** LOSS UNDER 0.05 - PROCESS FINISHED ***** \n")
            break


train(encoder, error, optimizer, 100, x)

with torch.no_grad():
    encoded = encoder.encode(x)
    decoded = encoder.decode(encoded)
    mse = error(decoded, x).item()
    enc = encoded.cpu().detach().numpy()
    dec = decoded.cpu().detach().numpy()

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(enc[:, 0], enc[:, 1], enc[:, 2], cmap=plt.cm.jet)

plt.title("ENCODED DATA: AST")
plt.show()
