from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tokengt.models import tokengt
import time

class TokenGT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = tokengt.TokenGTEncoder(args).half()
        self.embed_dim = args.encoder_embed_dim
        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        
        ## (노드->엣지) Graph encoder Pretrain 모델 load
        # COCO
        # checkpoint = torch.load('/data/data2/yjkim/coco_irra/base_logs/COCO/pretrain_model/best.pth') # 노드 -> 엣지 마스킹
        
        self.tokengt = TokenGT(args) 
        # self.tokengt.load_state_dict(checkpoint['model'])
        # # self.tokengt_proj = nn.Linear(1024, 768) 
        

        self.embed_dim = 1024
        # self.encoder = tokengt.TokenGTEncoder(args).half()

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            # self.cross_attn = nn.MultiheadAttention(self.embed_dim,
            #                                         self.embed_dim // 64,
            #                                         batch_first=True)
            # self.cross_modal_transformer = Transformer(width=self.embed_dim,
            #                                            layers=args.cmt_depth,
            #                                            heads=self.embed_dim //
            #                                            64)
            # scale = self.cross_modal_transformer.width**-0.5
            
            # self.ln_pre_t = LayerNorm(self.embed_dim)
            # self.ln_pre_i = LayerNorm(self.embed_dim)
            # self.ln_post = LayerNorm(self.embed_dim)

            # proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            # attn_std = scale
            # fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            # for block in self.cross_modal_transformer.resblocks:
            #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # # init cross attn
            # nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            # nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # # init mlm head
            # nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            # nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    # def cross_former(self, q, k, v):
    #     x = self.cross_attn(
    #             self.ln_pre_t(q),
    #             self.ln_pre_i(k),
    #             self.ln_pre_i(v),
    #             need_weights=False)[0]
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.cross_modal_transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD

    #     x = self.ln_post(x)
    #     return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()


    def forward(self, batch):
        ret = dict()

        # images = batch['images']
        # image_feats = self.base_model.encode_image(images)
        # i_feats = image_feats[:, 0, :].float()

        # graph_feats = self.base_model.encode_graph(batch)
        # g_feats = graph_feats[:,0,:].float()
        # logit_scale = self.logit_scale
        # ret.update({'temperature': 1 / logit_scale})

        # if 'itc' in self.current_task:
        #     ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        # if 'sdm' in self.current_task:
        #     ret.update({'sdm_loss':objectives.compute_sdm(i_feats, g_feats, batch['pids'], logit_scale)})

        # if 'cmpm' in self.current_task:
        #     ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        
        if 'mlm' in self.current_task:

            # mlm_feats = self.encoder(batch, mlm=True)
            mlm_feats = self.tokengt.encoder(batch, mlm=True)
            
            # m_feats = self.tokengt_proj(mlm_feats)
            
            # x = self.cross_former(mlm_feats, image_feats, image_feats)
            
            
            x = self.mlm_head(mlm_feats)  # [batch_size, text_len, num_colors]
            # x = self.mlm_head(m_feats)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_label'].reshape(-1)
            
            # score = x(학습된 값) / mlm_labels= 정답
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})
            

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        
        return ret  


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
