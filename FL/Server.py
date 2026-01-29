import numpy as np
import torch
from torch.autograd import Variable

from FL.client_lora import Client_LoRA
from FL.local_model_lora import Local_Model
from distill_dataset.pyramid_class_wise import DistillSubset


class Server_DF(object):

    def __init__(self,origin_model, inversion_model, client_num, task_num, subset, class_mask, device, test_data):
        self.origin_model=origin_model
        self.inversion_model=inversion_model

        self.client_num=client_num
        self.task_num=task_num
        self.subset=subset
        self.class_mask=class_mask

        self.device=device
        self.test_data=test_data

        self.clients = []
        self.task_id = -1

        self.global_model = Local_Model(self.origin_model)
        self.global_model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001, weight_decay=1e-03)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def init_client(self):
        print('Initialize clients')
        for i in range(0,1):
            # id,original_model,vit,task_per_global_epoch,subset,local_epoch,batch_size,lr,device,method
            # if i!=4:
            #     continue
            self.clients.append(Client_LoRA(i,self.origin_model, self.inversion_model, 1, self.subset[i], 40,8, 'cuda', self.class_mask[i],))
        print("Initialization completes")

    def train_clients(self):

        for t in range(self.task_num):
            for j in self.clients:
                j.train(round=t)
            self.task_id +=1
            self.train_global_model()





    def train_global_model(self):
        evaluate_class = []
        for i in range(self.client_num):
            for j in range(self.task_id+1):
                evaluate_class.extend(self.class_mask[i][j])

        temp_train_data = []
        temp_train_label = []
        for i in self.clients:
            data,label = i.distill_data.upload_Server()
            temp_train_data.append(data)
            temp_train_label.append(label)
        temp_train_data = torch.cat(temp_train_data)
        temp_train_label = torch.cat(temp_train_label)


        train_data = DistillSubset(temp_train_data,temp_train_label)
        train_loader_distill = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

        for epoch in range(200):
            for iteration, (input, target) in enumerate(train_loader_distill):
                input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda', non_blocking=True)
                logits = self.global_model(input)



                not_mask = np.setdiff1d(np.arange(200), evaluate_class)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                loss = self.criterion(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                self.scheduler.step()




    def evaluate_whole(self, task_id):
        evaluate_class = []
        for i in range(self.client_num):
            for j in range(task_id+1):
                evaluate_class.extend(self.class_mask[i][j])

        test_data = self.test_data.get_data(evaluate_class)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)


        self.global_model.eval()

        correct = 0
        total = 0

        for iteration, (input, target,_) in enumerate(test_loader):
            input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda',
                                                                                                                 non_blocking=True)

            output = self.global_model(input)
            # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
            # logits = pre

            # output_mixed = output['pre_logits']
            # pull_off = output['reduce_sim']

            # class_mask
            not_mask = np.setdiff1d(np.arange(200), evaluate_class)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
            logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'在{self.task_id}结束后，global model 的准确率为{acc}')

