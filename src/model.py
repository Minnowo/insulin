
import datetime

from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset
)
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

import pandas as pd
import numpy as np


class InsulinToCarbRatio(nn.Module):
    """
    InsulinToCarbRatio is designed to learn the insulinToCarbRatio based on the time of day.
    This ratio typically changes throughout the day:
        morning: 5
        lunch: 8
        dinner: 15
        night: 8

    So this model takes the hour of the day encoded as sin/cos and learns to predict the ratio.
    """

    def __init__(self, hidden=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, hidden),  # hour_sin, hour_cos
            nn.ReLU(),
            nn.Linear(hidden, 1),  
            nn.Softplus()          # ensures ICR stays >0
        )

    def forward(self, hour_sin: torch.Tensor, hour_cos: torch.Tensor):
        return self.net(torch.cat([hour_sin, hour_cos], dim=-1))

class InsulinCalculator(nn.Module):
    """
    InsulinCalculator uses the standard insulin calculation formula to predict an amount of insulin that needs to be taken.
    This model will hopefully learn the insulinSensitivityFactor and insulinToCarbRatio over time.
    """

    def __init__(self, insulinSensitivityFactor: float = 3, hiddenCarbRatio: int = 8):
        super().__init__()

        self.insulinSensitivityFactor = nn.Parameter(
            torch.tensor(insulinSensitivityFactor, dtype=torch.float32)
        )

        self.insulinToCarbRatio = InsulinToCarbRatio(hidden=hiddenCarbRatio)

    def forward(self,
                hour_sin: torch.Tensor,
                hour_cos: torch.Tensor,
                netCarbs: torch.Tensor,
                bloodGlucose: torch.Tensor,
                targetGlucose: torch.Tensor):

        insulinToCarbRatio = self.insulinToCarbRatio(hour_sin, hour_cos)

        dose = (netCarbs / insulinToCarbRatio) + ((bloodGlucose - targetGlucose) / self.insulinSensitivityFactor)

        return dose



class InsulinModule(LightningModule):
    def __init__(self, 
            learningRate: float = 0.005,
            insulinSensitivityFactor: float = 3, 
            hiddenCarbRatio: int = 8
        ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = learningRate

        self.net = InsulinCalculator(
            insulinSensitivityFactor=insulinSensitivityFactor,
            hiddenCarbRatio=hiddenCarbRatio
        )

        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_r2 = R2Score()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    def loss(self, y, target):
        return functional.mse_loss( y, target )

    def forward(self, batch_dict):
        return self.net(
            batch_dict["hour_sin"],
            batch_dict["hour_cos"],
            batch_dict["netCarbs"],
            batch_dict["bloodGlucose"],
            batch_dict["targetGlucose"]
        )

    def training_step(self, batch):

        batch_dict, target = batch

        target = target.view(-1)
        preds = self.forward(batch_dict).view(-1)

        loss = self.loss(preds, target)

        self.log("train_loss", loss, prog_bar=True)

        self.val_mae(preds, target)
        self.val_rmse(preds, target)
        self.val_r2(preds, target)

        self.log("train_mae", self.val_mae, prog_bar=True)
        self.log("train_rmse", self.val_rmse, prog_bar=True)
        self.log("train_r2", self.val_r2, prog_bar=True)

        return loss

    def validation_step(self, batch):

        batch_dict, target = batch

        target = target.view(-1)
        preds = self.forward(batch_dict).view(-1)

        self.val_mae(preds, target)
        self.val_rmse(preds, target)
        self.val_r2(preds, target)

        self.log("val_mae", self.val_mae, prog_bar=True)
        self.log("val_rmse", self.val_rmse, prog_bar=True)
        self.log("val_r2", self.val_r2, prog_bar=True)


class InsulinModule2(InsulinModule):

    def __init__(self, 
            learningRate: float = 0.005,
            minLR: float = 0.005,
            maxLR: float = 0.6,
            insulinSensitivityFactor: float = 3, 
            hiddenCarbRatio: int = 8
        ):
        super().__init__(learningRate=learningRate, insulinSensitivityFactor=insulinSensitivityFactor, hiddenCarbRatio=hiddenCarbRatio)
        self.save_hyperparameters()
        self.minLR = minLR
        self.maxLR = maxLR

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.minLR,
            max_lr=self.maxLR,
            step_size_up=500,
            mode="triangular2",
            cycle_momentum=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

class InsulinDataset(Dataset):

    def __init__(self, df: pd.DataFrame, targetCol: str = "insulinRec"):

        self.df = df.copy()
        self.targetCol = targetCol

        # encode the time using cos/sin because it cycles
        hours = self.df['dateTime'].dt.hour + self.df['dateTime'].dt.minute / 60
        self.df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        self.targets = torch.tensor(self.df['insulinTaken'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        batch_dict = {
            "hour_sin":  torch.tensor([row['hour_sin']], dtype=torch.float32),
            "hour_cos":  torch.tensor([row['hour_cos']], dtype=torch.float32),
            "netCarbs":  torch.tensor([row['netCarbs']], dtype=torch.float32),
            "bloodGlucose": torch.tensor([row['bloodGlucose']], dtype=torch.float32),
            "targetGlucose": torch.tensor([row['bloodGlucoseTarget']], dtype=torch.float32),
        }

        target = torch.tensor(row[self.targetCol], dtype=torch.float32)

        return batch_dict, target

def dataToTensorMap(dateTime: datetime.datetime, netCarbs: float, bloodGlucose: float, targetGlucose: float):

    hour = dateTime.hour + dateTime.minute / 60
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    data = {
        "hour_sin": torch.tensor([[hour_sin]], dtype=torch.float32),
        "hour_cos": torch.tensor([[hour_cos]], dtype=torch.float32),
        "netCarbs": torch.tensor([[netCarbs]], dtype=torch.float32),
        "bloodGlucose": torch.tensor([[bloodGlucose]], dtype=torch.float32),
        "targetGlucose": torch.tensor([[targetGlucose]], dtype=torch.float32)
    }

    return data
