import torch
import numpy as np
from  datetime import datetime

import model

def get_data(dateTime: datetime, netCarbs: float, bloodGlucose: float, targetGlucose: float):

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

def main():

    uma = model.InsulinModule.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=2-step=10185.ckpt")
    uma.eval()

    datas = [
        get_data(datetime(2025, 11, 5, 8, 40), 33.97, 8.4, 6.7),
        get_data(datetime(2025, 11, 5, 12, 1), 34.81, 5.3, 6.7),
    ]

    for data in datas:

        data = { key: value.to(uma.device) for key, value in data.items() }

        with torch.no_grad():

            predictions = uma(data)

            print(predictions)


if __name__ == "__main__":
    main()
