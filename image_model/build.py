import sys
import math
import utils
from typing import Iterable
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tokengt.models import tokengt
from mae.modeling_pretrain import PretrainVisionTransformer

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
        
        ## Masked image encoder (MAE)
        self.mae_image = PretrainVisionTransformer(args)

        ## Pretrained Image Encocer
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        
        # Pretrained Graph Encocer
        # graph_checkpoint = torch.load('/data/data2/khahn/coco_irra_phrase_masking/tmp_logs/COCO/20230831_071140_iira/best.pth') 

        # graph_checkpoint_model = {key.replace("tokengt.", ""): value for key, value in graph_checkpoint['model'].items()}
        self.tokengt = TokenGT(args) # TokenGT graph encoder
        # self.tokengt.load_state_dict(graph_checkpoint_model)
        
        # self.tokengt.load_state_dict(graph_checkpoint)
        
        self.tokengt_proj = nn.Linear(args.encoder_embed_dim, self.embed_dim) # 1024 -> 768

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True).to(dtype=torch.float16)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()


    def forward(self, batch):
        ret = dict()
        # device = torch.devcie

        images = batch['images'].half() # [64, 3, 224, 224]
        
        bool_masked_pos = batch['image_mask'].half()
        
        ## image mask
        bool_masked_pos = bool_masked_pos.to("cuda", non_blocking=True).flatten(1).to(torch.bool) # True/False 배열로 변환
        
    
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to("cuda")[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to("cuda")[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]
            patch_size = 16
            # if normlize_target(bool = True):
            images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
            images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            
            # we find that the mean is about 0.48 and standard deviation is about 0.08.
            images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            # else:
            #     images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            
            
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)  
  
        # caption_ids = batch['caption_ids']
        ## Unmasked Image Representation (encoded by CLIP pretrain image encoder)
        image_feats = self.base_model(images)  # [64, 197, 768]
        i_feats = image_feats[:, 0, :].float() # [64, 768]
        
        ## Masked Image Representation
        with torch.cuda.amp.autocast():
            masked_image_feats = self.mae_image(images, bool_masked_pos) # [64, 147, 768]
            # mse_loss_function = nn.MSELoss()
            # ret.update({'mse_loss': mse_loss_function(masked_image_feats, labels)})
            ret.update({'mse_loss':objectives.compute_mse(masked_image_feats, labels)})
  
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        return ret

def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
