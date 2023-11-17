import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

# from .coco_retrieval import * # DP
from .coco_retrieval_cs import * # CS
# from .rstpreid_retrieval import * # DP


def collate(batch):
    
    pids =  torch.stack([i['pids'] for i in batch])
    # image_ids = [i['image_ids'] for i in batch] # graph encoder에서는 이미지 필요없음
    # images = torch.stack([i['images'] for i in batch])

    caption_ids = torch.stack([i['caption_ids'] for i in batch])
    
    node_num = torch.tensor([i['node_data'].size(0) for i in batch])
    # edge_num = torch.tensor([i['edge_data'].size(0) for i in batch]) # CS
    max_n = max(node_num)

    edge_index = [i['edge_index'] for i in batch] # [2, sum(edge_num)] 
    edge_index = torch.cat(edge_index, dim=1) 

    # edge_data = [i['edge_data'] for i in batch] # [sum(edge_num), De] # CS 
    # edge_data = torch.cat(edge_data) # CS

    node_data = [i['node_data'] for i in batch] # [sum(node_num), Dn]
    node_data = torch.cat(node_data)

    # mlm_nodes = [i['mlm_nodes'] for i in batch] # [sum(node_num), Dn]
    # mlm_nodes = torch.cat(mlm_nodes)

    # mlm_edges = [i['mlm_edges'] for i in batch] # [sum(edge_num), De]
    # mlm_edges = torch.cat(mlm_edges)

    # [sum(node_num), Dl] = [sum(node_num), max_n]
    lap_eigvec = [i['lap_eigvec'] for i in batch]
    lap_eigvec = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigvec])

    lap_eigval = [i['lap_eigval'] for i in batch]
    lap_eigval = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigval])

    # sot_token = [i['sot_token'] for i in batch]
    # sot_token = torch.cat(sot_token)

    # eot_token = [i['eot_token'] for i in batch]
    # eot_token = torch.cat(eot_token)
    
    mlm_node = [i['mlm_node'] for i in batch]
    mlm_node = torch.cat(mlm_node)

    # mlm_edge = [i['mlm_edge'] for i in batch] # CS
    # mlm_edge = torch.cat(mlm_edge) # CS
    
    mlm_label = [i['mlm_label'] for i in batch]
    mlm_label = torch.stack(mlm_label)
    
    # node_info = [i['node_info'] for i in batch]
    # node_info = torch.tensor(node_info)
    
    # edge_info = [i['edge_info'] for i in batch]
    # edge_info = torch.tensor(edge_info)
    
    # node_edge_size = [i['node_edge_size'] for i in batch]
    # node_edge_size = torch.tensor(node_edge_size)
    
    # caption_size = [i['caption_size'] for i in batch]
    # caption_size = torch.tensor(caption_size)
    
    
    # mlm_pad = torch.zeros(1,node_num)
    # mlm_labels = [i['mlm_labels'] for i in batch]
    
    # mlm_labels = torch.cat(mlm_labels)

    item = dict(
        pids=pids,
        # images=images,
        caption_ids=caption_ids,
        edge_index=edge_index,
        # edge_data=edge_data, # CS
        node_data=node_data,
        lap_eigvec=lap_eigvec,
        lap_eigval=lap_eigval,
        node_num=node_num,
        # edge_num=edge_num, # CS
        # sot_token=sot_token,
        # eot_token=eot_token
        mlm_nodes=mlm_node,
        # mlm_edges=mlm_edge, # CS
        # mlm_ids=mlm_ids,
        mlm_label=mlm_label,
    )
    
    return item

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")
    num_workers = args.num_workers
    
 
    if args.MLM:
        train_set = RetrievalGraphDataset(args=args, 
                                        split='train',
                                        mlm=True)
        tmp=3
    else:
        train_set = RetrievalGraphDataset(args=args,
                                        split='train')

    val_set = RetrievalGraphEvalDataset(args=args,
                                    split='val')
    
    train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True, ########### 바꾸기!!
                                # shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=num_workers,
                            shuffle=False)

        
    return train_loader, val_loader

