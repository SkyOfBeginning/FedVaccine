import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from data.imagenet_r_subset_spliter import ImagenetR_spliter
from distill_dataset.pyramid_class_wise import log_images
from distill_dataset.pyramid_class_wise import PyramidDataset_Class_Wise
from models.get_models import get_model, get_fc
from FL.local_model import Local_Model
from timm.models import create_model, load_checkpoint
from utils_data import get_real_grad, get_syn_grad

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

inversion_model, num_features = get_model(name="dinov2_vitb", distributed=False)
# prompted_model = dinov2_vitb14(use_prompt = True, prompt_length = 10)

pretrained_cfg = create_model('vit_base_patch16_224').default_cfg

print(f"Creating my model:")
original_model = create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=200,
    # pretrained_cfg_overlay=dict(file='pretrain_model/pytorch_model.bin')
    # checkpoint_path='pretrain_model/original_model.pth'
    )

print(f"Creating model")
prompted_model = create_model(
    "vit_base_patch16_224",
    pretrained=False,
    pretrained_cfg=pretrained_cfg,
    num_classes=200,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
    prompt_length=10,
    # e_prompt_layer_idx=args.e_prompt_layer_idx,
    # method=args.method,
    # pretrained_cfg_overlay=dict(file='pretrain_model/pytorch_model.bin')
    # checkpoint_path='pretrain_model/model.pth'
    )

try:
    load_checkpoint(original_model, 'models/pretrain_model/ViT-B_16.npz')
    print("成功使用 load_checkpoint 加载并转换了 npz 权重。")
    load_checkpoint(prompted_model, 'models/pretrain_model/ViT-B_16.npz')
    print("成功使用 load_checkpoint 加载并转换了 npz 权重。")
except Exception as e:
    print(f"尝试使用 load_checkpoint 失败，错误: {e}")


for n, p in original_model.named_parameters():
    p.requires_grad = False

for n, p in prompted_model.named_parameters():
    p.requires_grad = False


client_data, client_mask = ImagenetR_spliter(client_num=1,task_num=20,input_size=224,private_class_num=200).random_split()

task_1_data = client_data[0][0]
task_1_mask = client_mask[0][0]

task_2_data = client_data[0][1]
task_2_mask = client_mask[0][1]

traindata, testdata = random_split(task_1_data, [int(len(task_1_data) * 0.7), len(task_1_data) - int(len(task_1_data) * 0.7)])
train_loader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=True)

local_model = Local_Model(original_model, prompted_model)
optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001,weight_decay=1e-03)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
criterion = torch.nn.CrossEntropyLoss().to('cuda')
local_model.to('cuda')
original_model.to('cuda')
prompted_model.to('cuda')
inversion_model.to('cuda')


for epoch in tqdm(range(25)):
    for iteration, (input, target,_) in enumerate(train_loader):
        input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda', non_blocking=True)

        output = local_model(input,0)
        # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
        # logits = pre
        logits = output['logits']
        # output_mixed = output['pre_logits']
        pull_off = output['reduce_sim']

        # class_mask
        # print(target)
        not_mask = np.setdiff1d(np.arange(200), task_1_mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)
        loss = loss - 0.05*pull_off
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()



##################  Evaluation ####################################
test_loader = DataLoader(testdata,batch_size=8,shuffle=True)
correct =0
total = 0


for iteration, (input, target, _) in enumerate(test_loader):
    input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda',non_blocking=True)

    output = local_model(input,0)
    # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
    # logits = pre
    logits = output['logits']
    # output_mixed = output['pre_logits']
    # pull_off = output['reduce_sim']

    # class_mask

    not_mask = np.setdiff1d(np.arange(200), task_1_mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))


    predicts = torch.max(logits, dim=1)[1].cpu()
    # print(predicts)
    # print(target)
    correct += (predicts == target.cpu()).sum()
    total += len(target)

acc = 100 * correct / total

print(f'任务1的准确率为{acc}')




################开始蒸馏###################


distill_data = PyramidDataset_Class_Wise(traindata,5)
AMP_SCALE = 10
fc = get_fc(num_feats=768, num_classes=200, distributed=False)
distill_epoch=2001

for h in task_1_mask:
    distill_data.set_pyramid_class(h)
    distill_data.init_optimizer()


    mydata = task_1_data.get_by_class(h)
    mydata_loader = DataLoader(mydata, batch_size=5, shuffle=True)
    AMP_SCALE = 10

    batch_real = None
    train_iter = iter(mydata_loader)

    for i in tqdm(range(distill_epoch)):
        # handling any synthetic data specific tasks such as adding another pyramid layer
        # can be used for other things if you add other representations
        distill_data.upkeep(step=i,Class=h)

        if i%500==0:
            distill_data.log_images(i,h)
        batch_real = next(train_iter, None)
        # reset the dataloader if we're at the end
        if batch_real is None:
            train_iter = iter(mydata_loader)
            batch_real = next(train_iter, None)

        x_real, y_real, _ = batch_real
        x_real = x_real.cuda(non_blocking=True)
        y_real = y_real.cuda(non_blocking=True)

        if i == 0:
            log_images(x_real, 00000, h)

        # get d_l_real / d_W
        grad_real = get_real_grad(x_real=x_real, y_real=y_real, model=inversion_model, fc=fc)

        # get synthetic images from pyramids
        x_syn, y_syn = distill_data.get_data()

        # get d_l_syn / d_W
        grad_syn = get_syn_grad(x_syn=x_syn, y_syn=y_syn, model=inversion_model, fc=fc)

        # print(grad_real)
        # print(grad_syn)
        # calculate meta loss as cosine distance between real and syn grads wrt W
        match_loss = 1 - torch.nn.functional.cosine_similarity(grad_real, grad_syn, dim=0)

        # we have to do manual grad scaling because of the second-order gradients
        match_loss *= AMP_SCALE

        output = local_model(x_syn, 0)
        # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
        # logits = pre
        logits = output['logits']

        if i >= 600:
            ce_loss = criterion(logits, y_syn)

            if i % 200 == 0:
                print(ce_loss, match_loss)

            total_loss = ce_loss + match_loss
        else:
            total_loss = match_loss
        # clear grads and backprop the meta loss
        distill_data.optimizer.zero_grad()
        total_loss.backward()
        distill_data.optimizer.step()
    distill_data.save_image()

    torch.cuda.empty_cache()




distill_train_data = distill_data.return_data()





# ###########################  第二个任务训练和回测第一个任务    ###############################
traindata2, testdata2 = random_split(task_2_data, [int(len(task_2_data) * 0.7), len(task_2_data) - int(len(task_2_data) * 0.7)])
train_loader2 = torch.utils.data.DataLoader(traindata2, batch_size=16, shuffle=True)

for epoch in tqdm(range(20)):
    for iteration, (input, target,_) in enumerate(train_loader2):
        input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda', non_blocking=True)

        output = local_model(input,1)
        # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
        # logits = pre
        logits = output['logits']
        # output_mixed = output['pre_logits']
        pull_off = output['reduce_sim']

        # class_mask

        not_mask = np.setdiff1d(np.arange(200), task_2_mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)
        loss = loss - 0.05*pull_off
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()



correct =0
total = 0


for iteration, (input, target, _) in enumerate(test_loader):
    input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda',non_blocking=True)

    output = local_model(input,0)
    # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
    # logits = pre
    logits = output['logits']
    # output_mixed = output['pre_logits']
    # pull_off = output['reduce_sim']

    # class_mask

    not_mask = np.setdiff1d(np.arange(200), task_1_mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))


    predicts = torch.max(logits, dim=1)[1].cpu()
    # print(predicts)
    # print(target)
    correct += (predicts == target.cpu()).sum()
    total += len(target)

acc = 100 * correct / total

print(f'任务1的准确率为{acc}')



#######################      使用蒸馏的数据继续训练   #########################################
train_loader_distill = torch.utils.data.DataLoader(distill_train_data, batch_size=5, shuffle=True)

for epoch in tqdm(range(300)):
    for iteration, (input, target) in enumerate(train_loader_distill):
        input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda', non_blocking=True)

        output = local_model(input,0)
        # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
        # logits = pre
        logits = output['logits']
        # output_mixed = output['pre_logits']
        pull_off = output['reduce_sim']

        # class_mask

        not_mask = np.setdiff1d(np.arange(200), task_1_mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)
        loss = loss - 0.05 * pull_off
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()



correct =0
total = 0


for iteration, (input, target, _) in enumerate(test_loader):
    input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda',non_blocking=True)

    output = local_model(input,0)
    # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
    # logits = pre
    logits = output['logits']
    # output_mixed = output['pre_logits']
    # pull_off = output['reduce_sim']

    # class_mask

    not_mask = np.setdiff1d(np.arange(200), task_1_mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))


    predicts = torch.max(logits, dim=1)[1].cpu()
    # print(predicts)
    # print(target)
    correct += (predicts == target.cpu()).sum()
    total += len(target)

acc = 100 * correct / total

print(f'任务1的准确率为{acc}')






