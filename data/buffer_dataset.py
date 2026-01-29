from typing import TypeVar, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class BufferDataset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], trans) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform_pretrain = trans



        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        # self.data = self.data
        self.targets = self.targets
        self.data = self.data
        self.transform = trans
        self.target_transform = None


    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_pre, target, idx


    def __len__(self):
        return len(self.data)


    def expand(self,newdataset,indexes):
        for i in indexes:
            self.data.append(newdataset.data[i])
            self.targets.append(newdataset.targets[i])





class MyTensorDataset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, buffer_feature_set) -> None:

        self.buffer_feature_set = buffer_feature_set


    def __getitem__(self, idx):

        img = self.buffer_feature_set[idx]



        return img


    def __len__(self):
        return len(self.buffer_feature_set)



