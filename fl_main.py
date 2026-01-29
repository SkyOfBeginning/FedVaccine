import random

import numpy as np
import torch
from timm import create_model
from timm.models import load_checkpoint

from FL.Server import Server_DF
from data.cifar100_subset_spliter import cifar100_Data_Spliter
from data.cub200_subset_spliter import CUB200_spliter
from data.imagenet_r_subset_spliter import ImagenetR_spliter
from models.get_models import get_model
from FL.vision_transformer_lora import VisionTransformer




inversion_model, num_features = get_model(name="dinov2_vitb", distributed=False)
# prompted_model = dinov2_vitb14(use_prompt = True, prompt_length = 10)

print('---------------------')
pretrained_cfg = create_model('vit_base_patch16_224').default_cfg


print(f"Creating my model:")
original_model = create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=200,
    # pretrained_cfg_overlay=dict(file='pretrain_model/pytorch_model.bin')
    # checkpoint_path='pretrain_model/original_model.pth'
    )
try:
    load_checkpoint(original_model, 'models/pretrain_model/ViT-B_16.npz')
    print("成功使用 load_checkpoint 加载并转换了 npz 权重。")
except Exception as e:
    print(f"尝试使用 load_checkpoint 失败，错误: {e}")




for n, p in original_model.named_parameters():
    p.requires_grad = False

inversion_model.to('cuda')
original_model.to('cuda')

seeds = [42,1999,2025]
# index = [[4,5],[6,7],[8,9],[10,11]]

for i in range(4):
    random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    client_data, client_mask = ImagenetR_spliter(client_num=5,task_num=5,input_size=224,private_class_num=40).random_split()

    test_data = ImagenetR_spliter(client_num=5,task_num=5,input_size=224,private_class_num=40).get_test_data()

    # client_data, client_mask = cifar100_Data_Spliter(client_num=5,task_num=5,input_size=224,private_class_num=15).random_split()
    # test_data = cifar100_Data_Spliter(client_num=5,task_num=5,input_size=224,private_class_num=15).get_test_data()

    # client_data, client_mask = CUB200_spliter(client_num=5,task_num=5,input_size=224,private_class_num=40).random_split()
    # test_data = CUB200_spliter(client_num=5,task_num=5,input_size=224,private_class_num=40).get_test_data()


    server = Server_DF(original_model, inversion_model, 5,5, client_data, client_mask, 'cuda', test_data)

    server.init_client()
    server.train_clients()

    print('\n \n \n')




