
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateFinder

import model

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def trainv0():
    seed_everything()

    df: pd.DataFrame = pd.read_csv("./data/home_insulin_clean_target_gc.csv", parse_dates=["dateTime"])

    split_idx = int(len(df) * 0.5)

    train_df = model.InsulinDataset(df.iloc[:split_idx])
    test_df  = model.InsulinDataset(df.iloc[split_idx:])

    train_loader = DataLoader( train_df, batch_size=50, shuffle=False, num_workers=4)
    test_loader = DataLoader( test_df, batch_size=50, shuffle=False, num_workers=4)

    uma = model.InsulinModule(learningRate=0.5, hiddenCarbRatio=16)
    trainer = Trainer(max_epochs=15, deterministic=True)
    trainer.fit(uma,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    print(trainer.validate(uma, test_loader))

    trainer.save_checkpoint("models/model_insulin_v0.ckpt")

def trainv1():
    seed_everything()

    df: pd.DataFrame = pd.read_csv("./data/home_insulin_clean_target_gc.csv", parse_dates=["dateTime"])

    split_idx = int(len(df) * 0.5)

    train_df = model.InsulinDataset(df.iloc[:split_idx], targetCol="insulinTaken")
    test_df  = model.InsulinDataset(df.iloc[split_idx:], targetCol="insulinTaken")

    train_loader = DataLoader( train_df, batch_size=50, shuffle=False, num_workers=4)
    test_loader = DataLoader( test_df, batch_size=50, shuffle=False, num_workers=4)

    uma = model.InsulinModule(learningRate=0.5, hiddenCarbRatio=16)
    trainer = Trainer(max_epochs=15, deterministic=True)
    trainer.fit(uma,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    print(trainer.validate(uma, test_loader))

    trainer.save_checkpoint("models/model_insulin_v1.ckpt")

def trainv2():
    seed_everything()

    df: pd.DataFrame = pd.read_csv("./data/home_insulin_clean_target_gc.csv", parse_dates=["dateTime"])

    split_idx = int(len(df) * 0.5)

    train_df = model.InsulinDataset(df.iloc[:split_idx])
    test_df  = model.InsulinDataset(df.iloc[split_idx:])

    train_loader = DataLoader( train_df, batch_size=5, shuffle=False, num_workers=4)
    test_loader = DataLoader( test_df, batch_size=5, shuffle=False, num_workers=4)

    uma = model.InsulinModule2(
            learningRate=0.5,
            minLR=0.0005,
            maxLR=1,
            hiddenCarbRatio=16)

    trainer = Trainer(max_epochs=15, deterministic=True)
    trainer.fit(uma,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    print(trainer.validate(uma, test_loader))

    trainer.save_checkpoint("models/model_insulin_v2.ckpt")


def main():

    print("Cuda avail: ", torch.cuda.is_available())
    print("Cuda device: ", torch.cuda.get_device_name(0))

    torch.manual_seed(0)
    trainv0()
    trainv1()
    trainv2()


if __name__ == "__main__":
    main()

