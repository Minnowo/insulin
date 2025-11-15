
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer

import torch
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset
)

import torch.nn as nn
import torch.nn.functional as functional

import torch
from torch.utils.data import Dataset
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

    def __init__(self, insulinSensitivityFactor: float = 3):
        super().__init__()

        self.insulinSensitivityFactor = nn.Parameter(
            torch.tensor(insulinSensitivityFactor, dtype=torch.float32)
        )

        self.insulinToCarbRatio = InsulinToCarbRatio()

    def forward(self,
                hour_sin: torch.Tensor,
                hour_cos: torch.Tensor,
                netCarbs: torch.Tensor,
                bloodGlucose: torch.Tensor,
                targetGlucose: torch.Tensor):

        insulinToCarbRatio = self.insulinToCarbRatio(hour_sin, hour_cos)

        dose = (netCarbs / insulinToCarbRatio) + ((bloodGlucose - targetGlucose) / self.insulinSensitivityFactor)

        return dose


class InsulinDataset(Dataset):

    def __init__(self, df: pd.DataFrame):

        self.df = df

        # encode the time using cos/sin because it cycles
        hours = self.df['dateTime'].dt.hour + self.df['dateTime'].dt.minute / 60
        self.df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        self.event_to_idx = {
            'breakfast': 0,
            'lunch':     1,
            'dinner':    2,
            'workout':   3
        }
        self.df['event_id'] = self.df['event'].map(self.event_to_idx).fillna(0)

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

        target = torch.tensor(row['insulinRec'], dtype=torch.float32)

        return batch_dict, target



class InsulinModule(LightningModule):
    def __init__(self):
        super().__init__()

        self.net = InsulinCalculator()
            
    def configure_optimizers(self):
        return optim.Adam(self.parameters())

    def loss(self, y, target):
        return functional.mse_loss(y, target)

    def forward(self, batch_dict):
        return self.net(
            batch_dict["hour_sin"],
            batch_dict["hour_cos"],
            batch_dict["netCarbs"],
            batch_dict["bloodGlucose"],
            batch_dict["targetGlucose"]
        )

    def training_step(self, batch, batch_index):

        batch_dict, target = batch

        preds = self.forward(batch_dict)

        loss = self.loss(preds, target)

        self.log("train_loss", loss, prog_bar=True)

        return loss

