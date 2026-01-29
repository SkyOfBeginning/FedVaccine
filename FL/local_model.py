from torch import nn
import torch

from FL.global_prompt import Global_Prompt


class Local_Model(nn.Module):

    def __init__(self, vit, prompted_vit, prompt_length=10, embed_dim=768, embedding_key='cls', prompt_init='uniform', prompt_pool=True, prompt_key=True, pool_size=10, top_k=1, batchwise_prompt=True, prompt_key_init='uniform', dropout=0.0):
        super().__init__()


        self.dropout = nn.Dropout(dropout)
        self.vit = vit
        self.prompted_vit = prompted_vit

        self.head = nn.Sequential(nn.Linear(768, 200),
                                 )

        self.prompt = Global_Prompt(length=prompt_length, embed_dim=embed_dim, embedding_key=embedding_key,
                                    prompt_init=prompt_init,
                                    prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k,
                                    batchwise_prompt=batchwise_prompt,
                                    prompt_key_init=prompt_key_init)

        self.prompted_vit.set_prompt(self.prompt)

        self.heads = []



    def forward(self, x,task_id):
        # with torch.no_grad():
        #     temp = self.vit(x)
        #     cls_features = temp['pre_logits']
        #
        # output = self.prompted_vit.forward_features_with_prompt(x, None, self.prompt, cls_features)
        #
        # output['logits'] = self.head(output['feat'])


        with torch.no_grad():
            output = self.vit(x)
            cls_features = output['pre_logits']

        output = self.prompted_vit(x, cls_features=cls_features,train=True,task_id=task_id)
        output['logits'] = self.head(output['feat'])


        return output


    def save_heads(self):
        temp = self.head.state_dict()
        self.heads.append(temp)

    def load_heads(self,task_id):
        self.head.load_state_dict(self.heads[task_id])

















