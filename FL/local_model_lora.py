import math

from torch import nn
import torch

from FL.global_prompt import Global_Prompt


class Local_Model(nn.Module):

    def __init__(self, vit, r=4, alpha=16, dropout=0.0,index =[2,3],):
        super().__init__()


        self.dropout = nn.Dropout(dropout)
        self.vit = vit


        self.head = nn.Sequential(nn.Linear(768, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 200))

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        orig_linear = self.vit.blocks[0].attn.qkv
        self.orig = orig_linear
        self.index= index

        self.A = nn.Parameter(torch.randn((self.r,self.orig.in_features)))
        self.Q = nn.Parameter(torch.randn((self.orig.out_features, self.r)))

        print(self.A.numel())
        print(self.Q.numel())
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.Q)



        self.sig_value = {}
        self.heads = []
        self.previous_lora = []



    def forward(self, x):
        x = self.vit.forward_features_with_LoRA(x, self.A, self.Q, self.index, self.scaling)
        x = self.head(x)

        return x


    def forward_feat(self, x):
        x = self.vit.forward_features_with_LoRA(x, self.A, self.Q, self.index, self.scaling)
        return x




    def load_heads(self,task_id):
        self.head.load_state_dict(self.heads[task_id])
        self.head.to('cuda')

    def save_heads(self):
        self.heads.append(self.head.state_dict())


    def decompose_sig_value(self,task_id):
        with torch.no_grad():
            delta_W = self.Q @ self.A
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        r = self.r
        Sigma_task = S[:r]  # (r,)
        V_task = Vh[:r].T  # (d_in, r)

        # print(Sigma_task.shape)
        # print(V_task.shape)

        self.sig_value[task_id] = (Sigma_task.detach(), V_task.detach())
        self.previous_lora.append((self.A,self.Q))



    def spectral_recall_loss(self, task_id, eps=1e-8):
        """
        A:           (r, d_in)
        Q:           (d_out, r)
        V_task:      (d_in, r)
        Sigma_task:  (r,)
        """
        Sigma_task, V_task = self.sig_value[task_id]

        delta_W = self.Q @ self.A  # (d_out, d_in)

        # 投影到 task 子空间
        proj = delta_W @ V_task  # (d_out, r)

        # 奇异值加权
        Sigma_sqrt = torch.sqrt(Sigma_task + eps)  # (r,)

        loss = torch.norm(proj * Sigma_sqrt[None, :],p='fro') ** 2

        return loss



    def re_init(self, task_id):
        """
                A:           (r, d_in)
                Q:           (d_out, r)
                V_task:      (d_in, r)
                Sigma_task:  (r,)
                """

        self.A, self.Q= self.previous_lora[task_id]




















