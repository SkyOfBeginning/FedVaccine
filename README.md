# FedVaccine

This repository contains PyTorch implementation code.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 4080
- Python 3.8

## Usage
First, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
pillow==9.2.0
matplotlib==3.5.3
```

## Pretrain models
Our method  loads pre-trained ViT locally. You can remove the following two lines of code from main.py to switch to online loading:
```
pretrained_cfg = create_model(args.model).default_cfg
pretrained_cfg['file']='pretrain_model/ViT-B_16.npz'
```



## Training
To train a model via command line:

Single node with single gpu


```
python fl_main.py 
```
In fl_main.py, you can switch dataset by cifar100_Data_Spliter(5, 5, 15, 224).random_split() or ImagenetR_spliter(5, 5, 40, 224).random_split()





## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

