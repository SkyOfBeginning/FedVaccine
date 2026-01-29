import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from distill_dataset.pyramid_class_wise import PyramidDataset_Class_Wise, log_images
from models.get_models import get_fc
from FL.local_model import Local_Model

from utils_data import get_real_grad, get_syn_grad


class Client_DF(object):

    def __init__(self,id,original_model, prompted_model, inversion_model, task_per_global_epoch, subset, local_epoch, batch_size, lr, device, class_mask):
        self.id = id
        self.original_model = original_model
        self.prompted_model = prompted_model
        self.inversion_model = inversion_model

        self.class_mask = class_mask
        self.task_id = -1
        self.task_per_global_epoch = task_per_global_epoch
        self.test_loader=[]
        # subset应该是一个【】，其中包含了num_task个数据以及类别，以[[(类别)：[数据]]，{}]的形式保存
        self.train_data =subset
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device=device







    def init_local_model(self):

        self.local_model = Local_Model(self.original_model, self.prompted_model)
        self.distill_data = PyramidDataset_Class_Wise(5)
        self.criterion = torch.nn.CrossEntropyLoss().to('cuda')
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001, weight_decay=1e-03)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.local_epoch)

        self.local_model.to('cuda')
        self.original_model.to('cuda')
        self.prompted_model.to('cuda')
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

                output = self.local_model(input, self.task_id)
                # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
                # logits = pre
                logits = output['logits']
                # output_mixed = output['pre_logits']
                pull_off = output['reduce_sim']

                # class_mask
                # print(target)
                not_mask = np.setdiff1d(np.arange(200), self.class_mask[self.task_id])
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target)
                loss = loss - 0.05 * pull_off
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                self.scheduler.step()

        self.distillation(self.task_id)
        self.evaluate(self.task_id)

        ################# tnse 可视化     ##########################

        if self.task_id== 0:
            self.local_model.eval()
            ori_features = []
            ori_labels = []

            with torch.no_grad():
                for x, y in tqdm(train_loader):
                    x, y = Variable(x, requires_grad=False).to('cuda', non_blocking=True), y.long().to('cuda', non_blocking=True)

                    # ⚠️ 根据你的模型改这里
                    # 假设 model(x) 返回 feature, logits
                    output = self.local_model(x, self.task_id)
                    feat = output['feat']
                    # 如果 model(x) 返回的是 (feat, logits)

                    ori_features.append(feat.cpu())
                    ori_labels.append(y)

            ori_features = torch.cat(ori_features, dim=0).numpy()
            ori_labels = torch.cat(ori_labels, dim=0).numpy()

            recall_features = []
            recall_labels = []
            distill_train_data = self.distill_data.get_by_task(0)
            train_loader_distill = torch.utils.data.DataLoader(distill_train_data, batch_size=8, shuffle=True)
            with torch.no_grad():
                for iteration, (input, target) in enumerate(train_loader_distill):
                    input, target = Variable(input, requires_grad=False).to('cuda',non_blocking=True), target.long().to('cuda', non_blocking=True)


                    output = self.local_model(input, self.task_id)
                    feat = output['feat']
                    # 如果 model(x) 返回的是 (feat, logits)

                    recall_features.append(feat.cpu())
                    recall_labels.append(target)

            recall_features = torch.cat(recall_features, dim=0).numpy()
            recall_labels = torch.cat(recall_labels, dim=0).numpy()

            tsne = TSNE(n_components=2,perplexity=30,n_iter=1000,random_state=42,init='pca',learning_rate='auto')

            tsne_results = tsne.fit_transform(ori_features)
            recall_results = tsne.fit(recall_features)
            plt.figure(figsize=(8, 8))
            num_classes = len(np.unique(ori_labels))

            for cls in self.class_mask[0]:
                idx = ori_labels == cls
                plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], s=10, label=str(cls), alpha=0.7, marker='o')

            plt.legend(markerscale=2, fontsize=9)
            plt.xticks([])
            plt.yticks([])
            plt.title("t-SNE Visualization of Features")
            plt.show()
            plt.savefig('t-SNE Visualization of Features.pdf', dpi=300, bbox_inches='tight')




        ###################################################################################################
        if self.task_id > 0:
            print('回忆前：')
            for i in range(self.task_id + 1):
                self.evaluate(i)

            print('---------------------------------------------------------------')
            print('回忆后')
            for i in range(self.task_id + 1):
                self.recall(i)

            print('#####################################')




    def distillation(self,task):
        AMP_SCALE = 10
        fc = get_fc(num_feats=768, num_classes=200, distributed=False)
        distill_epoch = 2001
        for h in tqdm(self.class_mask[task]):
            self.distill_data.set_pyramid_class(h)
            self.distill_data.init_optimizer()

            mydata = self.train_dataset.get_by_class(h)
            mydata_loader = DataLoader(mydata, batch_size=5, shuffle=True)

            batch_real = None
            train_iter = iter(mydata_loader)

            for i in range(distill_epoch):
                # handling any synthetic data specific tasks such as adding another pyramid layer
                # can be used for other things if you add other representations
                self.distill_data.upkeep(step=i, Class=h)

                # if i % 500 == 0:
                #     self.distill_data.log_images(i, h)
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

                output = self.local_model(x_syn, task)
                # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
                # logits = pre
                logits = output['logits']

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
        train_loader_distill = torch.utils.data.DataLoader(distill_train_data, batch_size=8, shuffle=True)

        for epoch in tqdm(range(300)):
            for iteration, (input, target) in enumerate(train_loader_distill):
                input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to(
                    'cuda', non_blocking=True)

                output = self.local_model(input, task)
                # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
                # logits = pre
                logits = output['logits']
                # output_mixed = output['pre_logits']
                pull_off = output['reduce_sim']

                # class_mask

                not_mask = np.setdiff1d(np.arange(200), self.class_mask[task])
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = self.criterion(logits, target)
                loss = loss - 0.05 * pull_off
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                self.scheduler.step()

        self.evaluate(task)






    def evaluate(self, task=0):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data,batch_size=8,shuffle=True)
        correct =0
        total = 0

        for iteration, (input, target, _) in enumerate(test_loader):
            input, target = Variable(input, requires_grad=False).to('cuda', non_blocking=True), target.long().to('cuda',
                                                                                                                 non_blocking=True)

            output = self.local_model(input, task)
            # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
            # logits = pre
            logits = output['logits']
            # output_mixed = output['pre_logits']
            # pull_off = output['reduce_sim']

            # class_mask
            not_mask = np.setdiff1d(np.arange(200), self.class_mask[task])
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'客户端{self.id}\t任务{task}的准确率为\t{acc}')








