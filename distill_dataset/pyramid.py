import os
import random
from typing import List, Tuple, T_co

import numpy as np
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset

from .base import BaseDistilledDataset

def log_images(syn_images: Tensor, step: int):

    grid = torchvision.utils.make_grid(
        syn_images, normalize=False, scale_each=False, nrow=5
    )
    # wandb.log({"grids/raw": wandb.Image(grid)}, step=step)
    # 假设 syn_images 在 [-1, 1] 范围内，需要转换为 [0, 1]
    grid_np = grid.cpu().numpy()

    # 调整通道顺序：matplotlib 期望 (H, W, C)
    grid_np = np.transpose(grid_np, (1, 2, 0))

    # 3. 绘制并保存
    plt.figure(figsize=(13, 13))
    plt.imshow(grid_np)
    plt.axis('off')

    # 保存到本地文件
    save_path = os.path.join('best_images/', f"raw_grid_step_{step}.png")
    plt.savefig(save_path)
    print(f"图像已保存到: {save_path}")

    plt.close()  # 关闭图形，释放内存


class PyramidDataset(BaseDistilledDataset):

    def __init__(self, train_dataset,num):

        super().__init__()
        self.train_dataset = train_dataset

        self.num_per_class = num



    def init_optimizer(self):
        optimizer = torch.optim.Adam([{"params": p, "lr": 2e-3} for p in self.pyramid])
        self.optimizer = optimizer
        return optimizer

    def init_synset(self,class_mask) -> Tuple[List[Tensor], Tensor]:
        print(class_mask)
        total_num = len(class_mask)*self.num_per_class
        syn_labels = torch.cat([torch.tensor([h] * self.num_per_class, dtype=torch.long) for h in class_mask],dim=0).cuda()

        print("syn_label is")
        print(syn_labels)
        perm = torch.randperm(syn_labels.size(0), device=syn_labels.device)
        # 2. 使用随机索引打乱 syn_labels
        syn_labels = syn_labels[perm]
        print(syn_labels)

        num_images = total_num

        pyramid = []
        res = 1

        while res <= 1:
            level = torch.randn((num_images, 3, res, res), device="cuda")
            if "noise" == "zero":
                level = level * 0
            pyramid.insert(0, level)
            res *= 2

            # to make it work when res is not power of 2
            if res > 224:
                res = 224

        pyramid = [p / len(pyramid) for p in pyramid]

        for p in pyramid:
            p.requires_grad_(True)

        self.pyramid, self.syn_labels = pyramid, syn_labels


    # def set_pyramid_class(self,target_class):
    #     class_mask = (self.syn_labels_multi == target_class)
    #     self.pyramid = self.pyramids[class_mask]
    #     self.syn_labels = self.syn_labels_multi[class_mask]


    def extend_pyramid(self) -> bool:

        print("extending pyramid...")

        old_len = len(self.pyramid)
        new_len = len(self.pyramid) + 1

        old_res = self.pyramid[0].shape[-1]

        if old_res == 256:
            print("already max res")
            return False
        else:
            new_res = old_res * 2
            # to make it work when res is not power of 2
            if new_res > 224:
                new_res = 224
            print("new res: {}".format(new_res))

        num_images = self.pyramid[-1].shape[0]

        self.pyramid = [p.detach().clone() * old_len / new_len for p in self.pyramid]
        if "noise" == "zero":
            new_layer = torch.sum(
                torch.stack(
                    [
                        torch.nn.functional.interpolate(
                            p, (new_res, new_res), antialias=False, mode="bilinear"
                        )
                        for p in self.pyramid
                    ]
                ),
                dim=0,
            )
            new_layer = new_layer / old_len
        else:
            new_layer = (
                torch.randn((num_images, 3, new_res, new_res), device="cuda") / new_len
            )

        self.pyramid.insert(0, new_layer)

        for p in self.pyramid:
            p.requires_grad_(True)

        self.optimizer = self.init_optimizer()

        return True

    def decode_pyramid(self) -> Tensor:

        result = torch.sum(
            torch.stack(
                [
                    torch.nn.functional.interpolate(
                        p,
                        (224, 224),
                        antialias=False,
                        mode="bilinear",
                    )
                    for p in self.pyramid
                ]
            ),
            dim=0,
        )


        result = self.linear_decorrelate_color(result)

        result = torch.sigmoid(2 * result)

        return result

    def get_data(self) -> Tuple[Tensor, Tensor]:
        syn_images = self.decode_pyramid()
        return syn_images, self.syn_labels


    def reture_dataset(self):
        syn_images = self.decode_pyramid()
        return DistillSubset(syn_images,self.syn_labels)
    @torch.no_grad()
    def log_images(self, step: int = None):
        if len(self.pyramid[0]) > 100:
            print("Warning: too many images to log")
            return
        with torch.no_grad():

            syn_images, _ = self.get_data()
            syn_images = syn_images.detach().clone()

            log_images(syn_images=syn_images, step=step)

    def upkeep(self, step: int = None):
        if (step - 1) % 200 == 0 and step > 1:
            if self.extend_pyramid():
                self.log_images(step=step)

    def get_save_dict(self):
        save_dict = {
            "pyramid": self.pyramid,
            "opt_state": self.optimizer.state_dict(),
        }
        return save_dict

    def load_from_dict(self, load_dict: dict):

        loaded_pyramid = load_dict["pyramid"]
        while len(self.pyramid) < len(loaded_pyramid):
            self.extend_pyramid()

        with torch.no_grad():
            for p, loaded_p in zip(self.pyramid, loaded_pyramid):
                p.copy_(loaded_p)

        self.optimizer.load_state_dict(load_dict["opt_state"])



class DistillSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, data,target) -> None:
        self.data = data
        self.target = target


    def __getitem__(self, idx):
        img, target = self.data[idx], self.target[idx]

        # Return image, target, original index, and the buffer flag
        return img, target

    def __len__(self):
        return len(self.data)
