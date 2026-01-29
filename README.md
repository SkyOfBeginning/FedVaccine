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


If the cuda memory is over-flowing or the speed is too slow, please lower the value on Line 211 in utils_data.py:
```angular2html
x_syn = syn_augmentor(torch.cat([x_syn for a in range(10)]))

# if GPU memory is over-flowing, reduce num(10) to num(2)
y_syn = torch.cat([y_syn for a in range(10)])
```

The default method is task-wise vaccine generation, you can change it into class-wise  vaccine generation in client_lora.py
```angular2html
# you can use class-wise pyramidDataset or just task-wise pyramidDataset
# self.distill_data = PyramidDataset_Class_Wise(3)
self.distill_data =PyramidDataset_Task_Wise(3)
```




## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

