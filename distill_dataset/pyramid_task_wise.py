import os
import random
from typing import List, Tuple, T_co

from torchvision.transforms import transforms

from augmentation.ops import RandomGaussianNoise, RandomHorizontalFlip, RandomResizedCrop
import numpy as np
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import Dataset

from .base import BaseDistilledDataset

def log_images(syn_images: Tensor, step: int, Class: int):
    pass
    # grid = torchvision.utils.make_grid(
    #     syn_images, normalize=False, scale_each=False, nrow=5
    # )
    # # wandb.log({"grids/raw": wandb.Image(grid)}, step=step)
    # # 假设 syn_images 在 [-1, 1] 范围内，需要转换为 [0, 1]
    # grid_np = grid.cpu().numpy()
    #
    # # 调整通道顺序：matplotlib 期望 (H, W, C)
    # grid_np = np.transpose(grid_np, (1, 2, 0))
    #
    # # 3. 绘制并保存
    # plt.figure(figsize=(13, 13))
    # plt.imshow(grid_np)
    # plt.axis('off')
    #
    # # 保存到本地文件
    # save_path = os.path.join('class_wise_best_images/moco/', f"{Class}_raw_grid_step_{step}.png")
    # plt.savefig(save_path)
    # # print(f"图像已保存到: {save_path}")
    #
    # plt.close()  # 关闭图形，释放内存


class PyramidDataset_Task_Wise(BaseDistilledDataset):

    def __init__(self,num):

        super().__init__()


        self.num_per_class = num

        self.all_images = []
        self.all_labels = []

        self.task_data = {}



    def init_optimizer(self):
        optimizer = torch.optim.Adam([{"params": p, "lr": 2e-3} for p in self.pyramid])
        self.optimizer = optimizer
        return optimizer

    def init_synset(self,target_class) -> Tuple[List[Tensor], Tensor]:

        total_num = self.num_per_class
        syn_labels = torch.cat([torch.tensor(target_class * self.num_per_class, dtype=torch.long)],dim=0).cuda()

        perm = torch.randperm(syn_labels.size(0), device=syn_labels.device)
        # 2. 使用随机索引打乱 syn_labels
        syn_labels = syn_labels[perm]

        num_images = len(syn_labels)
        print(num_images)

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


    def set_pyramid_class(self,target_class):
        self.init_synset(target_class)
        self.init_optimizer()

    def save_image(self):
        syn_images = self.decode_pyramid()
        self.all_images.append(syn_images)
        self.all_labels.append(self.syn_labels)


    def extend_pyramid(self) -> bool:

        # print("extending pyramid...")

        old_len = len(self.pyramid)
        new_len = len(self.pyramid) + 1

        old_res = self.pyramid[0].shape[-1]

        if old_res == 224:
            # print("already max res")
            return False
        else:
            new_res = old_res * 2
            # to make it work when res is not power of 2
            if new_res > 224:
                new_res = 224
            # print("new res: {}".format(new_res))

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


    def return_data(self):
        all_image = torch.cat(self.all_images, dim=0)
        all_label = torch.cat(self.all_labels, dim=0)
        return DistillSubset(all_image, all_label)

    def save_per_task(self,task):
        self.task_data[task] = (torch.cat(self.all_images, dim=0),torch.cat(self.all_labels, dim=0))
        self.all_images.clear()
        self.all_labels.clear()

    def get_by_task(self,task):
        image,label = self.task_data[task]
        return DistillSubset(image,label)

    def upload_Server(self):
        image = torch.cat([self.task_data[h][0] for h in self.task_data.keys()], dim=0)
        label = torch.cat([self.task_data[h][1] for h in self.task_data.keys()], dim=0)
        return image,label


    @torch.no_grad()
    def log_images(self, step: int = None, Class: int = None) -> Tensor:
        if len(self.pyramid[0]) > 100:
            print("Warning: too many images to log")
            return
        with torch.no_grad():

            syn_images, _ = self.get_data()
            syn_images = syn_images.detach().clone()

            log_images(syn_images=syn_images, step=step, Class = Class)

    def upkeep(self, step: int = None):
        if (step - 1) % 200 == 0 and step > 1:
            if self.extend_pyramid():
                self.log_images(step=step,Class=None)

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
        self.trans = transforms.RandomHorizontalFlip(p=0.5)



    def __getitem__(self, idx):
        img, target = self.data[idx], self.target[idx]

        # Return image, target, original index, and the buffer flag
        return self.trans(img), target

    def __len__(self):
        return len(self.data)
