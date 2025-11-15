
import pandas as pd

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer

import model

def main():

    print("Cuda avail: ", torch.cuda.is_available())
    print("Cuda device: ", torch.cuda.get_device_name(0))

    df: pd.DataFrame = pd.read_csv("./data/home_insulin_clean_target_gc.csv", parse_dates=["dateTime"])

    split_idx = int(len(df) * 0.5)

    train_df = model.InsulinDataset(df.iloc[:split_idx])
    test_df  = model.InsulinDataset(df.iloc[split_idx:])

    train_loader = DataLoader(
        train_df, 
        batch_size=32, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_df, 
        batch_size=32, 
        shuffle=False
    )

    uma = model.InsulinModule()

    trainer = Trainer(max_epochs=3, deterministic=True)
    trainer.fit(uma,
                train_dataloaders=train_df,
                val_dataloaders=test_df)


if __name__ == "__main__":
    main()

