import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from fastestimator.dataset.dataset import DatasetSummary, FEDataset
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list

@traceable()
class PartDataset(FEDataset):
    def __init__(self, dataset: Union[FEDataset, Iterable[FEDataset]], rate: float) -> None:
        self.dataset = dataset
        self.rate = rate
        self.orig_len = len(self.dataset)
        self.samples = math.ceil(self.orig_len * self.rate)

        self.index_maps_full = list(range(self.orig_len))
        random.shuffle(self.index_maps_full)
        self.index_ptr = 0
        self.index_maps = None
        self.reset_index_maps()

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        if isinstance(index, int):
            return self.dataset[self.index_maps[index]]

    def reset_index_maps(self) -> None:
        #
        if self.index_ptr + self.samples <= self.orig_len:
            self.index_maps = self.index_maps_full[self.index_ptr:self.index_ptr + self.samples]
            self.index_ptr = self.index_ptr + self.samples
            if self.index_ptr == self.orig_len:
                random.shuffle(self.index_maps_full)
                self.index_ptr = 0

        else:
            first_part = self.index_maps_full[self.index_ptr:]  # take self.orig_len - self.index_ptr
            random.shuffle(self.index_maps_full)
            second_part = self.index_maps_full[:self.samples + self.index_ptr -
                                               self.orig_len]  # take self.samples + self.index_ptr - self.orig_len
            self.index_maps = first_part + second_part
            self.index_ptr = self.samples + self.index_ptr - self.orig_len