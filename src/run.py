import torch
import numpy as np
from  datetime import datetime

import model

def main():

    uma = model.InsulinModule.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=2-step=10185.ckpt")
    uma.eval()

    datas = [
        model.dataToTensorMap(datetime(2025, 11, 5, 8, 40), 33.97, 8.4, 6.7),
        model.dataToTensorMap(datetime(2025, 11, 5, 12, 1), 34.81, 5.3, 6.7),
        model.dataToTensorMap(datetime(2025, 11, 5, 18, 0), 66, 10.3, 6.7),
        model.dataToTensorMap(datetime(2025, 11, 5, 21, 57), 16, 6.3, 6.7),
    ]

    for data in datas:

        data = { key: value.to(uma.device) for key, value in data.items() }

        with torch.no_grad():

            predictions = uma(data)

            print(predictions)


if __name__ == "__main__":
    main()
