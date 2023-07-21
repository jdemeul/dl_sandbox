import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import pandas as pd


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.x = np.arange(0, 10, 0.1)
        self.y = 2 * self.x + 1

    def setup(self, stage=None):
        self.x = torch.from_numpy(self.x).float().unsqueeze(-1)
        self.y = torch.from_numpy(self.y).float().unsqueeze(-1)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.x, self.y), batch_size=8)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.x, self.y), batch_size=8)

mydata = pd.read_csv(filepath_or_buffer="./tab.txt")
print(mydata.shape)
print(mydata.loc[mydata["header2"] > 1,"header3"])

thismodel = MyModel()

accel = "cpu"
if torch.backends.mps.is_available():
    accel = "mps"
if torch.cuda.is_available():
    accel = "cuda"

print(accel)