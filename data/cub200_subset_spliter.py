import random
from typing import TypeVar, Sequence

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

from data.continual_datasets import CUB200
from data.imagenet_r_subset_spliter import Test_set
from utils import build_transform

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class CUB200_spliter():

    def __init__(self,client_num,task_num,private_class_num,input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        self.private_class_num = private_class_num
        self.input_size = input_size



    # 分成client_num数目个subset,每个subset里包含了task个subsubset
    def random_split(self):
        trans = build_transform(True,self.input_size)
        self.Imagenet_R = CUB200(root='D:/datasets/local_datasets', train=True, download=True)
        trainset = self.Imagenet_R

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

        # 分类
        total_private_class_num = self.client_num*self.private_class_num
        public_class_num = 200-total_private_class_num
        class_public = [i for i in range(200)]
        class_public = set(class_public)
        class_public = list(class_public)

        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        class_private = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0,self.client_num)]
            # random.shuffle(class_private[i])
        # print(class_private)


        # 对每个客户端进行操作
        client_subset = [[] for i in range(0,self.client_num)]
        client_mask = [[] for i in range(0,self.client_num)]
        surro_index = []

        class_every_task = int((public_class_num+self.private_class_num)/self.task_num)
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # while  (a < 0.1).any():
            #     a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]

        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_private[i][j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_private[i][j*class_every_task:j*class_every_task+class_every_task]:
                    if k in class_public:
                        # 是公共类
                        lenth = int(int(class_counts[k])*dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        q = 0
                        while q < lenth:
                            random_index = random.choice(list(unused_indice))
                            index.append(random_index)
                            unused_indice.remove(random_index)
                            q += 1
                        class_label[k]=unused_indice
                    else: #是私有类
                        index.extend(class_label[k])
                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset,index,trans,None))
                # if 0<=i and i <=4:
                #     client_mask[i+5].append(class_this_task)
                #     client_subset[i+5].append(Vicious_CustomedSubset(trainset, index, trans))


        # for i in tqdm(range(200)):
        #     q = 0
        #     unused_indice = set(class_label[i])
        #
        #     while q < surrogate_num:
        #         random_index = random.choice(list(unused_indice))
        #         surro_index.append(random_index)
        #         unused_indice.remove(random_index)
        #         q += 1
        #
        # surrodata = CustomedSubset(trainset,surro_index,trans,None)

        return client_subset,client_mask


    def process_testdata(self,surrogate_num):
        trans = build_transform(False,self.input_size)
        self.Imagenet_R_test = CUB200(root='./local_datasets', train=False, download=True)
        testset = self.Imagenet_R_test
        # 100个类别的数据分给三个客户端使用

        class_counts = torch.zeros(200)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        surro_index =[]
        test_index = []
        for i in tqdm(range(200)):
            q = 0
            unused_indice = set(class_label[i])

            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index.extend(list(unused_indice))
        surrodata = CustomedSubset(testset,surro_index,trans,None)
        testdata = CustomedSubset(testset,test_index,trans,None)
        return surrodata,testdata

    def random_split_synchron(self):
        trans = build_transform(False,self.input_size)
        self.Imagenet_R = CUB200_spliter(root='D:/datasets/local_datasets', train=True, download=True)
        trainset = self.Imagenet_R

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
        # 分类

        class_public = [i for i in range(200)]
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
                client_subset[i].append(CustomedSubset(trainset, index, trans,None))

                if 0<=i and i <=4:
                    client_mask[i+5].append(class_this_task)
                    client_subset[i+5].append(Vicious_CustomedSubset(trainset, index, trans))

        return client_subset,client_mask
    def random_split_10_task(self):
        trans = build_transform(True,self.input_size)
        self.Imagenet_R = CUB200(root='D:/datasets/local_datasets', train=True, download=True)
        trainset = self.Imagenet_R

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


        # 分类

        class_public = [i for i in range(200)]

        class_p = random.sample(class_public, 150)
        class_public = list(set(class_public) - set(class_p))

        class_private = []
        for i in range(self.client_num):
            class_pri = random.sample(class_p,50)
            class_pri = class_pri + class_public
            random.shuffle(class_pri)
            class_private.append(class_pri)



        # 对每个客户端进行操作
        client_subset = [[] for i in range(0,self.client_num)]
        client_mask = [[] for i in range(0,self.client_num)]

        class_every_task = 10


        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_private[i][j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_private[i][j*class_every_task:j*class_every_task+class_every_task]:
                    # 是公共类
                    a = random.uniform(0.2, 0.9)
                    lenth = int(int(class_counts[k])*a)
                    unused_indice = set(class_label[k])

                    q = 0
                    while q < lenth:
                        random_index = random.choice(list(unused_indice))
                        index.append(random_index)
                        unused_indice.remove(random_index)
                        q += 1

                random.shuffle(index)

                client_subset[i].append(CustomedSubset(trainset,index,trans,None))

        return client_subset,client_mask


    def random_split_DLG(self):
        trans = build_transform(True,self.input_size)
        self.cifar100_dataset =CUB200(root='D:/datasets/local_datasets', train=True, download=True)
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

        index1 = []
        index2 = []
        for k in range(16):
            if k<8:
                leng = len(class_label[k])
                unused_indice = set(class_label[k])
                q = 0
                while q < leng:
                    random_index = random.choice(list(unused_indice))
                    index1.append(random_index)
                    unused_indice.remove(random_index)
                    q += 1
                class_label[k]=unused_indice
            elif k>=8:
                leng = len(class_label[k])
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
        class_mask2 = [i for i in range(8,16)]


        return CustomedSubset(trainset,index1,trans,None), class_mask1 ,CustomedSubset(trainset,index2,trans,None),class_mask2



    def random_split_continual_learning(self):
        trans = build_transform(True,self.input_size)
        self.cifar100_dataset = CUB200(root='D:/datasets/local_datasets', train=True, download=True)
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
        # +1是恶意客户端
        class_mask = []
        subsets = []


        for j in range(0,20):
            index = []
            class_this_task = [i for i in range(j*10,j*10+10)]
            class_mask.append(class_this_task)
            for k in class_this_task:
                # print(k)
                leng = len(class_label[k])
                unused_indice = set(class_label[k])
                q = 0
                while q < leng:
                    random_index = random.choice(list(unused_indice))
                    index.append(random_index)
                    unused_indice.remove(random_index)
                    q += 1
                class_label[k] = unused_indice

            random.shuffle(index)

            subsets.append(CustomedSubset(trainset,index,trans,None))




        return subsets,class_mask

    def get_test_data(self):
        # CUB200(root='D:/datasets/local_datasets', train=False, download=True).split()
        self.Imagenet_R_test = CUB200(root='D:/datasets/local_datasets', train=False, download=True)
        testset = self.Imagenet_R_test
        # print(testset.data)
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

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], trans, show_sample) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform_pretrain = trans

        self.show_sample = show_sample

        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        # self.data = self.data
        self.targets = np.array(self.targets)
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
        my_trans = build_transform(False,224)
        self.transform_pretrain = my_trans

    def get_by_class(self,target_class):
        # 1. 查找对应类别的索引
        # 使用布尔索引：self.targets == target_class 会返回一个布尔数组
        class_mask = (self.targets == target_class)
        class_idx = np.where(class_mask)[0]  # 取出 True 的位置
        class_data = [self.data[i] for i in class_idx]
        # 2. 提取数据和标签
        # 使用布尔数组作为索引，直接从 self.data 和 self.targets 中选择匹配的元素
        # class_data = self.data[class_mask]
        class_targets = self.targets[class_mask]

        # 3. 返回结果
        return subsubset(class_data, class_targets,self.transform)







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
        self.data = self.data
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
        return img_pre,target

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








