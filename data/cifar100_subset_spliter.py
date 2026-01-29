import random
import time
from typing import TypeVar, Sequence

import numpy as np

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets import MNIST,CIFAR100
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

from data.imagenet_r_subset_spliter import Test_set
from utils_data import build_transform

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class cifar100_Data_Spliter():

    def __init__(self,client_num,task_num,private_class_num,input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        self.transform1 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.private_class_num = private_class_num
        self.input_size = input_size



    # 分成client_num数目个subset,每个subset里包含了task个subsubset
    def random_split(self):
        trans = build_transform(True,self.input_size)
        self.cifar100_dataset = CIFAR100(root='D:/datasets/local_datasets', train=True, download=True)
        trainset = self.cifar100_dataset

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(100) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        # 分类
        total_private_class_num = self.client_num*self.private_class_num
        print(total_private_class_num)
        public_class_num = 100-total_private_class_num
        class_public = [i for i in range(100)]
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        class_private = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0,self.client_num)]
        for i in range(0,self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])


        # 对每个客户端进行操作
        # +1是恶意客户端
        client_subset = [[] for i in range(0,self.client_num+5)]
        client_mask = [[] for i in range(0,self.client_num+5)]

        class_every_task = int((public_class_num+self.private_class_num)/self.task_num)
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            while  (a < 0.1).any():
                a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]
        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_private[i][j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_private[i][j*class_every_task:j*class_every_task+class_every_task]:
                    if k in class_public:
                        # 是公共类
                        len = int(int(class_counts[k])*dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        q = 0
                        while q < len:
                            random_index = random.choice(list(unused_indice))
                            index.append(random_index)
                            unused_indice.remove(random_index)
                            q += 1
                        class_label[k]=unused_indice
                    else: #是私有类
                        index.extend(class_label[k])
                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset,index,trans))
                if 0<=i and i <=4:
                    client_mask[i+5].append(class_this_task)
                    client_subset[i+5].append(Vicious_CustomedSubset(trainset, index, trans))




        return client_subset,client_mask


    def process_testdata(self,surrogate_num):
        trans = build_transform(False,self.input_size)
        self.cifar100_dataset = CIFAR100(root='D:/datasets/local_datasets', train=False, download=True)
        testset = self.cifar100_dataset
        # 100个类别的数据分给三个客户端使用

        class_counts = torch.zeros(100)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        surro_index =[]
        test_index = []
        for i in tqdm(range(100)):
            q = 0
            unused_indice = set(class_label[i])
            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index.extend(list(unused_indice))
        surrodata = CustomedSubset(testset,surro_index,trans)
        testdata = CustomedSubset(testset,test_index,trans)
        return surrodata,testdata

    def random_split_synchron(self):
        trans = build_transform(False,self.input_size)
        self.cifar100_dataset = CIFAR100(root='D:/datasets/local_datasets', train=True, download=True)
        trainset = self.cifar100_dataset

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(100) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        # 分类
        class_public = [i for i in range(100)]


        # 对每个客户端进行操作
        client_subset = [[] for i in range(0,self.client_num+5)]
        client_mask = [[] for i in range(0,self.client_num+5)]

        class_every_task = 10
        dirichlet_perclass = {}

        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_public[j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                # 是公共类
                for k in class_this_task:
                    len = int(int(class_counts[k]) * random.uniform(0.3, 0.85))
                    unused_indice = set(class_label[k])
                    q = 0
                    while q < len:
                        random_index = random.choice(list(unused_indice))
                        index.append(random_index)
                        unused_indice.remove(random_index)
                        q += 1

                    random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset,index,trans))
                if 0<=i and i <=4:
                    client_mask[i+5].append(class_this_task)
                    client_subset[i+5].append(Vicious_CustomedSubset(trainset, index, trans))

        return client_subset,client_mask



    def random_split_DLG(self):
        trans = build_transform(True,self.input_size)
        self.cifar100_dataset =CIFAR100(root='D:/datasets/local_datasets', train=True, download=True)
        trainset = self.cifar100_dataset

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(200) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index




        # 对每个客户端进行操作

            # 对每个客户端进行操作

        index1 = []
        index2 = []
        for k in range(16):
            if k < 8:
                leng = len(class_label[k])*0.5
                unused_indice = set(class_label[k])
                q = 0
                while q < leng:
                    random_index = random.choice(list(unused_indice))
                    index1.append(random_index)
                    unused_indice.remove(random_index)
                    q += 1
                class_label[k] = unused_indice
            elif k >= 8:
                leng = len(class_label[k])*0.5
                unused_indice = set(class_label[k])
                q = 0
                while q < leng:
                    random_index = random.choice(list(unused_indice))
                    index2.append(random_index)
                    unused_indice.remove(random_index)
                    q += 1
                class_label[k] = unused_indice

        random.shuffle(index1)
        random.shuffle(index2)

        class_mask1 = [i for i in range(8)]
        class_mask2 = [i for i in range(8, 16)]

        return CustomedSubset(trainset, index1, trans), class_mask1, CustomedSubset(trainset, index2, trans,), class_mask2



    def random_split_continual_learning(self):
        trans = build_transform(True,self.input_size)
        self.cifar100_dataset = CIFAR100(root='D:/datasets/local_datasets', train=True, download=True)
        trainset = self.cifar100_dataset

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(100) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index




        # 对每个客户端进行操作
        # +1是恶意客户端
        class_mask = []
        subsets = []


        for j in range(0,10):
            index = []
            class_this_task = [i for i in range(j*10,j*10+10)]
            class_mask.append(class_this_task)
            for k in class_this_task:
                # print(k)
                leng = len(class_label[k]) * 0.5
                unused_indice = set(class_label[k])
                q = 0
                while q < leng:
                    random_index = random.choice(list(unused_indice))
                    index.append(random_index)
                    unused_indice.remove(random_index)
                    q += 1
                class_label[k] = unused_indice

            random.shuffle(index)

            subsets.append(CustomedSubset(trainset,index,trans))




        return subsets,class_mask

    def get_test_data(self):
        self.Imagenet_R_test = CIFAR100(root='D:/datasets/local_datasets', train=False, download=True)
        testset = self.Imagenet_R_test

        return Test_set(testset)




class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],trans) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform_pretrain = trans
        self.transform_origin = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.target_transform = None
        self.transform = trans

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_pre, target, idx

    def get_by_class(self,target_class):
        # 1. 查找对应类别的索引
        # 使用布尔索引：self.targets == target_class 会返回一个布尔数组
        class_mask = (self.targets == target_class)

        # 2. 提取数据和标签
        # 使用布尔数组作为索引，直接从 self.data 和 self.targets 中选择匹配的元素
        class_data = self.data[class_mask]
        class_targets = self.targets[class_mask]

        # 3. 返回结果
        return subsubset(class_data, class_targets,self.transform)


    def get_test_sample(self):
        img = self.show_sample

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)

        return img

    def __len__(self):
        return len(self.indices)

    def switch_train_transform(self):
        self.transform_pretrain = self.transform

    def switch_target_transform(self):
        my_trans = build_transform(False, 224)
        self.transform_pretrain = my_trans



class Vicious_CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],trans) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform_pretrain = trans
        self.transform_origin = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.target_transform = None

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = self.data[random.choice(range(len(self.data)))]
        img = Image.fromarray(img)

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)


        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_pre, target

    def __len__(self):
        return len(self.indices)



class subsubset(Dataset[T_co]):
    def __init__(self, data, target, trans) -> None:


        self.data = data
        self.targets = target

        self.transform_pretrain = trans
        self.transform_origin = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform = trans

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)



        return img_pre, target, idx


    def __len__(self):
        return len(self.targets)









