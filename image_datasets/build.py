import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from utils.comm import get_world_size

from .coco_retrieval import *
# from .rstpreid_retrieval import *

def build_transforms(img_size=(224, 224), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    
    pids =  torch.stack([i['pids'] for i in batch]) # [64]

    images = torch.stack([i['images'] for i in batch]) # [64, 3, 224, 224]
    caption_ids = torch.stack([i['caption_ids'] for i in batch])
   
    node_num = torch.tensor([i['node_data'].size(0) for i in batch]) # [64]
    # edge_num = torch.tensor([i['edge_data'].size(0) for i in batch])
    max_n = max(node_num)

    edge_index = [i['edge_index'] for i in batch] # [2, sum(edge_num)]
    edge_index = torch.cat(edge_index, dim=1) # [2, 3606]

    # edge_data = [i['edge_data'] for i in batch] # [sum(edge_num), De]
    # edge_data = torch.cat(edge_data)

    node_data = [i['node_data'] for i in batch] # [sum(node_num), Dn]
    node_data = torch.cat(node_data) # [1867]

    # [sum(node_num), Dl] = [sum(node_num), max_n]
    lap_eigvec = [i['lap_eigvec'] for i in batch]
    lap_eigvec = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigvec]) # [1867, 44]

    lap_eigval = [i['lap_eigval'] for i in batch]
    lap_eigval = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigval]) # [1867, 44]

    
    mlm_node = [i['mlm_node'] for i in batch]
    mlm_node = torch.cat(mlm_node) # [1867]

    # mlm_edge = [i['mlm_edge'] for i in batch]
    # mlm_edge = torch.cat(mlm_edge)
    
    mlm_label = [i['mlm_label'] for i in batch]
    mlm_label = torch.stack(mlm_label) # [64, 77]
    
    
    # image_mask = [torch.stack([i['image_mask'] for i in batch]) # [batch_size, num_patches]=[64, 196]
    image_mask = [i['image_mask'] for i in batch]
    image_mask = torch.cat(image_mask) # [64, 196]


    item = dict(
        pids=pids,
        images=images,
        caption_ids=caption_ids,
        edge_index=edge_index,
        # edge_data=edge_data,
        node_data=node_data,
        lap_eigvec=lap_eigvec,
        lap_eigval=lap_eigval,
        node_num=node_num,
        # edge_num=edge_num,
        mlm_nodes=mlm_node,
        # mlm_edges=mlm_edge,
        mlm_label=mlm_label,
        image_mask = image_mask
    )
    
    return item

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")
    num_workers = args.num_workers
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            train_set = RetrievalGraphDataset(args=args, 
                                            split='train',
                                            transforms=train_transforms,
                                            mlm=True)
        else:
            train_set = RetrievalGraphDataset(args=args,
                                         split='train',
                                         transforms=train_transforms)

        val_set = RetrievalGraphEvalDataset(args=args,
                                       split='val',
                                       transforms=val_transforms)
        
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                # shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate)
        val_loader = DataLoader(val_set,
                                # batch_size=args.test_batch_size,
                                batch_size=args.batch_size,
                                num_workers=num_workers,
                                shuffle=False)

        t=3
        return train_loader, val_loader

