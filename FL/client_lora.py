import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from distill_dataset import PyramidDataset
from distill_dataset.pyramid_class_wise import PyramidDataset_Class_Wise, log_images
from distill_dataset.pyramid_task_wise import PyramidDataset_Task_Wise
from models.get_models import get_fc
from FL.local_model_lora import Local_Model

from utils_data import get_real_grad, get_syn_grad
import umap

class Client_LoRA(object):

    def __init__(self,id,original_model,  inversion_model, task_per_global_epoch, subset, local_epoch,batch_size, device, class_mask):
        self.id = id
        self.original_model = original_model

        self.inversion_model = inversion_model
        self.batch_size = batch_size

        self.class_mask = class_mask
        self.task_id = -1
        self.task_per_global_epoch = task_per_global_epoch
        self.test_loader=[]
        # subset应该是一个【】，其中包含了num_task个数据以及类别，以[[(类别)：[数据]]，{}]的形式保存
        self.train_data =subset
        self.local_epoch = local_epoch


        self.device=device

        self.init_local_model()








    def init_local_model(self):

        self.local_model = Local_Model(self.original_model)

        # you can use class-wise pyramidDataset or just task-wise pyramidDataset
        # self.distill_data = PyramidDataset_Class_Wise(3)
        self.distill_data =PyramidDataset_Task_Wise(2)
        self.criterion = torch.nn.CrossEntropyLoss().to('cuda')
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001, weight_decay=1e-03)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.local_epoch)

        self.local_model.to('cuda')
        self.original_model.to('cuda')
        self.inversion_model.to('cuda')



    def get_data(self,task_id):
        self.train_dataset = self.train_data[task_id]
        self.current_class = self.class_mask[task_id]
        print(f'{self.id} client，{task_id} task has {len(self.current_class)} classes:{self.current_class}')
        trainset = self.train_dataset
        traindata, testdata = random_split(trainset,[int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])

        self.test_loader.append(testdata)

        self.traindata = traindata
        print(len(traindata))


    def train(self, round):
        if round%self.task_per_global_epoch==0:
            self.task_id = self.task_id + 1
            self.get_data(self.task_id)

        criterion = torch.nn.CrossEntropyLoss().to('cuda')
        train_loader = DataLoader(self.traindata, batch_size=self.batch_size, shuffle=True)

        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input, target, _) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda', non_blocking=True)

                logits = self.local_model(input)

                # class_mask
                # print(target)
                not_mask = np.setdiff1d(np.arange(200), self.class_mask[self.task_id])
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target)
                loss = loss
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                self.scheduler.step()

        self.local_model.decompose_sig_value(self.task_id)
        if self.task_id == 0:
            self.distillation(self.task_id)
        self.local_model.save_heads()
        self.evaluate(self.task_id)


        if self.task_id == 4:
            print(f'{self.id}客户端回忆前和回忆后：')
            for i in range(self.task_id):
                self.evaluate(i)

            print('---------------------------------------------------------------')

            # for i in range(self.task_id):
            #     self.recall(i)

            self.recall(0)


            print('#####################################')
        time.sleep(3)




    def distillation(self,task):
        AMP_SCALE = 10
        fc = get_fc(num_feats=768, num_classes=200, distributed=False)
        distill_epoch = 2001

        # class_wise
        # for h in tqdm(self.class_mask[task]):
        #     self.distill_data.set_pyramid_class(h)
        #     self.distill_data.init_optimizer()
        #
        #     mydata = self.train_dataset.get_by_class(h)
        #     mydata_loader = DataLoader(mydata, batch_size=5, shuffle=True)
        #
        #     batch_real = None
        #     train_iter = iter(mydata_loader)
        #
        #     for i in range(distill_epoch):
        #         # handling any synthetic data specific tasks such as adding another pyramid layer
        #         # can be used for other things if you add other representations
        #         self.distill_data.upkeep(step=i, Class=h)
        #
        #         if i % 500 == 0:
        #             self.distill_data.log_images(i, h)
        #         batch_real = next(train_iter, None)
        #         # reset the dataloader if we're at the end
        #         if batch_real is None:
        #             train_iter = iter(mydata_loader)
        #             batch_real = next(train_iter, None)
        #
        #         x_real, y_real, _ = batch_real
        #         x_real = x_real.cuda(non_blocking=True)
        #         y_real = y_real.cuda(non_blocking=True)
        #
        #         if i == 0:
        #             log_images(x_real, 00000, h)
        #
        #         # get d_l_real / d_W
        #         grad_real = get_real_grad(x_real=x_real, y_real=y_real, model=self.inversion_model, fc=fc)
        #
        #         # get synthetic images from pyramids
        #         x_syn, y_syn = self.distill_data.get_data()
        #
        #         # get d_l_syn / d_W
        #         grad_syn = get_syn_grad(x_syn=x_syn, y_syn=y_syn, model=self.inversion_model, fc=fc)
        #
        #         # print(grad_real)
        #         # print(grad_syn)
        #         # calculate meta loss as cosine distance between real and syn grads wrt W
        #         match_loss = 1 - torch.nn.functional.cosine_similarity(grad_real, grad_syn, dim=0)
        #
        #         # we have to do manual grad scaling because of the second-order gradients
        #         match_loss *= AMP_SCALE
        #
        #         output = self.local_model(x_syn)
        #         not_mask = np.setdiff1d(np.arange(200), self.class_mask[self.task_id])
        #         not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
        #         logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
        #         # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
        #         # logits = pre
        #         # logits = output['logits']
        #
        #         if i >= 600:
        #             ce_loss =  self.criterion(logits, y_syn)
        #             total_loss = ce_loss + match_loss
        #         else:
        #             total_loss = match_loss
        #         # clear grads and backprop the meta loss
        #         self.distill_data.optimizer.zero_grad()
        #         total_loss.backward()
        #         self.distill_data.optimizer.step()
        #     self.distill_data.save_image()
        #     torch.cuda.empty_cache()
        # self.distill_data.save_per_task(task)





        self.distill_data.set_pyramid_class(self.class_mask[task])
        self.distill_data.init_optimizer()

        mydata = self.train_dataset
        mydata_loader = DataLoader(mydata, batch_size=8, shuffle=True)

        batch_real = None
        train_iter = iter(mydata_loader)

        for i in tqdm(range(distill_epoch)):
            # handling any synthetic data specific tasks such as adding another pyramid layer
            # can be used for other things if you add other representations
            self.distill_data.upkeep(step=i)

            if i % 500 == 0:
                self.distill_data.log_images(i, None)
            batch_real = next(train_iter, None)
            # reset the dataloader if we're at the end
            if batch_real is None:
                train_iter = iter(mydata_loader)
                batch_real = next(train_iter, None)

            x_real, y_real, _ = batch_real
            x_real = x_real.cuda(non_blocking=True)
            y_real = y_real.cuda(non_blocking=True)


            # get d_l_real / d_W
            grad_real = get_real_grad(x_real=x_real, y_real=y_real, model=self.inversion_model, fc=fc)

            # get synthetic images from pyramids
            x_syn, y_syn = self.distill_data.get_data()

            # get d_l_syn / d_W
            grad_syn = get_syn_grad(x_syn=x_syn, y_syn=y_syn, model=self.inversion_model, fc=fc)

            # print(grad_real)
            # print(grad_syn)
            # calculate meta loss as cosine distance between real and syn grads wrt W
            match_loss = 1 - torch.nn.functional.cosine_similarity(grad_real, grad_syn, dim=0)

            # we have to do manual grad scaling because of the second-order gradients
            match_loss *= AMP_SCALE

            output = self.local_model(x_syn)
            not_mask = np.setdiff1d(np.arange(200), self.class_mask[self.task_id])
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
            logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
            # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
            # logits = pre
            # logits = output['logits']

            if i >= 600:
                ce_loss =  self.criterion(logits, y_syn)
                total_loss = ce_loss + match_loss
            else:
                total_loss = match_loss
            # clear grads and backprop the meta loss
            self.distill_data.optimizer.zero_grad()
            total_loss.backward()
            self.distill_data.optimizer.step()
        self.distill_data.save_image()
        torch.cuda.empty_cache()
        self.distill_data.save_per_task(task)


    def recall(self, task):
        distill_train_data = self.distill_data.get_by_task(task)
        self.local_model.re_init(task)
        # self.local_model.load_heads(task)
        train_loader_distill = torch.utils.data.DataLoader(distill_train_data, batch_size=8, shuffle=True)
        optimizer = torch.optim.Adam([
            {"params": self.local_model.A, "lr": 1e-6},  # e.g. 1e-4 or 5e-5
            {"params": self.local_model.Q, "lr": 1e-6},
            {"params": self.local_model.head.parameters(),"lr": 1e-3}], weight_decay=1e-03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        self.local_model.cuda()
        for epoch in range(20):
            for iteration, (input, target) in enumerate(train_loader_distill):
                input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda', non_blocking=True)

                logits = self.local_model(input)


                # class_mask

                not_mask = np.setdiff1d(np.arange(200), self.class_mask[task])
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))


                loss = self.criterion(logits, target) + 0.6 * self.local_model.spectral_recall_loss(task)
                # if epoch % 100 ==0 :
                #     print(loss, recall_loss)

                loss = loss
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()

        self.evaluate(task)







    def evaluate(self, task=0):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data,batch_size=8,shuffle=True)
        correct =0
        total = 0
        # self.local_model.load_heads(task)
        self.local_model.eval()
        for iteration, (input, target, _) in enumerate(test_loader):
            input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda',
                                                                                                                 non_blocking=True)

            output = self.local_model(input)
            # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
            # logits = pre

            # output_mixed = output['pre_logits']
            # pull_off = output['reduce_sim']

            # class_mask
            not_mask = np.setdiff1d(np.arange(200), self.class_mask[task])
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
            logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')










