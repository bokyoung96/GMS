import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorchtools import *
from cs_analysis import *

with open('res_adj.pkl', 'rb') as f:
    data = pickle.load(f)
    print("===== DATA LOADED =====")

scaler = RobustScaler()


class CustomerDL(CustomerAnalysis):
    def __init__(self, item_type: str = 'ast'):
        super().__init__(item_type)

    def get_raw_data(self):
        strs_mkts = self.finder_cols(self.strs_mkts)
        strs_trs = self.finder_cols(self.strs_trs)
        strs_ast = self.finder_cols(self.strs_ast)
        strs_cols = strs_mkts + strs_trs + strs_ast

        res = data.drop(strs_cols, axis=1)
        print("===== TEMP DATA (W/O STRINGS) LOADED =====")
        return res

    def get_preprocess_data(self):
        strs_data = self.get_raw_data()
        strs_data_mean = strs_data.mean()
        strs_data = strs_data.fillna(strs_data_mean)
        temp = pd.DataFrame(scaler.fit_transform(strs_data),
                            columns=strs_data.columns)
        res = torch.Tensor(temp.to_numpy())
        return res

    def get_train_test_split_data(self, batch_size: int = 256, test_size: float = 0.2, random_state: int = 42):
        temp = self.get_preprocess_data()
        x_train, x_test = train_test_split(
            temp, test_size=test_size, random_state=random_state)
        train_loader = DataLoader(x_train, batch_size=batch_size)
        valid_loader = DataLoader(x_train, batch_size=batch_size)
        test_loader = DataLoader(x_test, batch_size=batch_size)
        return train_loader, valid_loader, test_loader

# PARAMETERS


batch_size = 256
test_size = 0.2
random_state = 42

input_size = len(data.columns)
hidden_sizes = [164, 82, 41, 20, 10, 3]
learning_rate = 0.001

train_loader, valid_loader, test_loader = CustomerDL(
).get_train_test_split_data(batch_size, test_size, random_state)


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, hidden_size))
            encoder_layers.append(nn.ReLU())
            prev_size = hidden_size
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for hidden_size in reversed(hidden_sizes[:-1]):
            decoder_layers.append(nn.Linear(prev_size, hidden_size))
            decoder_layers.append(nn.ReLU())
            prev_size = hidden_size

        decoder_layers.append(nn.Linear(prev_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_size, hidden_sizes).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(model)


def train_model(model, batch_size, patience, n_epochs):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch)
            # calculate the loss
            loss = criterion(output, batch)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, data)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses


# batch_size = 256
# n_epochs = 500

# # early stopping patience; how long to wait after last time validation loss improved.
# patience = 20

# model, train_loss, valid_loss = train_model(
#     model, batch_size, patience, n_epochs)
