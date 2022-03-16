from matplotlib.font_manager import json_dump
import numpy as np
from torch.utils.data import IterableDataset
from typing import Iterator, T_co


class ClimateHackDataset(IterableDataset):

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        seqSize = 36
        dataLen = self.dataset.shape[0]
        seqNum = int(dataLen/seqSize)
        fts = []
        for k in range(4):
            for j in range(4):
                for i in range(seqNum):
                    data = self.dataset[i:i+seqSize]
                    xco1 = 128*k
                    xco2 = 128*(k+1)
                    yco1 = 128*j
                    yco2 = 128*(j+1)
                    features = data[:12, yco1:yco2, xco1:xco2]
                    targets = data[12:, yco1+32:yco2-32, xco1+32:xco2-32]
                    ft = [features, targets]
                    fts.append(ft)
        self.dataset = fts

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.dataset)
