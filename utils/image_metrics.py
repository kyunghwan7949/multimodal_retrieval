from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import time
import datetime
from tqdm import tqdm
from utils.meter import AverageMeter



class Evaluator():
    def __init__(self, data_loader):
        # self.img_loader = img_loader # gallery
        # self.txt_loader = txt_loader # query
        self.data_loader = data_loader
        self.logger = logging.getLogger("IRRA.eval")
    
    @torch.no_grad()
    def eval(self, model):
        
        logger = logging.getLogger("IRRA.eval")
        logger.info('start evaluation')

        tmp = model.mae_image
        
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        logging.info("Computing features for evaluation...")
        start_time = time.time()

        images = self.data_loader.dataset.graph
        num_image = len(images)
        image_bs = 64

        for i in range(0, num_image, image_bs):
            # batch_i

            image = images[i : min(num_image, i + image_bs)]
            batch_image = self.collate(image)
            batch = {k: v.to(device) for k, v in batch_image.items()}
            ret = model(batch)

        return ret
    
    def collate(self, batch):
        batch = batch
        # pids =  torch.stack([i['pids'] for i in batch]) # [64]
        
        images = torch.stack([i['images'] for i in batch]) # [64, 3, 224, 224]
        caption_ids = torch.stack([i['caption_ids'] for i in batch])
        
        node_num = torch.tensor([i['node_data'].size(0) for i in batch])
        # edge_num = torch.tensor([i['edge_data'].size(0) for i in batch])
        max_n = max(node_num)

        edge_index = [i['edge_index'] for i in batch] # [2, sum(edge_num)]
        edge_index = torch.cat(edge_index, dim=1)

        # # edge_data = [i['edge_data'] for i in batch] # [sum(edge_num), De]
        # # edge_data = torch.cat(edge_data)

        node_data = [i['node_data'] for i in batch] # [sum(node_num), Dn]
        node_data = torch.cat(node_data)

        # # [sum(node_num), Dl] = [sum(node_num), max_n]
        lap_eigvec = [i['lap_eigvec'] for i in batch]
        lap_eigvec = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigvec])

        lap_eigval = [i['lap_eigval'] for i in batch]
        lap_eigval = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigval])

        
        mlm_node = [i['mlm_node'] for i in batch]
        mlm_node = torch.cat(mlm_node)

        # # mlm_edge = [i['mlm_edge'] for i in batch]
        # # mlm_edge = torch.cat(mlm_edge)
        
        mlm_label = [i['mlm_label'] for i in batch]
        mlm_label = torch.stack(mlm_label)

        # pids =  torch.stack([i['pids'] for i in batch]) # [64]

        image_mask = [i['image_mask'] for i in batch]
        image_mask = torch.cat(image_mask) # [64, 196]
        
        item = dict(
            # pids=pids,
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


    # @torch.no_grad()
    # def compute_sim_ma ix(self, model, data_loader):
        
    #     # model.float()
    #     model.eval()
    #     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #     logging.info("Computing features for evaluation...")
    #     start_time = time.time()

    #     texts = data_loader.dataset.text
    #     num_text = len(texts)
    #     text_bs = 256
    #     text_features = []

    #     for i in range(0, num_text, text_bs):

    #         text = texts[i : min(num_text, i + text_bs)]
    #         text_input = torch.stack(text).to(device)
    #         text_feat = model.encode_text(text_input).float()
    #         text_feat = F.normalize(text_feat, dim=-1)
    #         text_features.append(text_feat)

    #     text_features = torch.cat(text_features, dim=0)


    #     image_features = []
    #     for samples in tqdm(data_loader):
    #         image = samples["image"]

    #         image = image.to(device)
    #         image_feat = model.encode_image(image).float()
    #         image_feat = F.normalize(image_feat, dim=-1)
    #         image_features.append(image_feat)

    #     image_features = torch.cat(image_features, dim=0)
        
    #     sims_matrix_i2t = image_features @ text_features.t()
    #     sims_matrix_t2i = sims_matrix_i2t.t()

    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     logging.info("Evaluation time {}".format(total_time_str))
        
    #     return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy()

    # @torch.no_grad()
    # def report_metrics(self, scores_i2t, scores_t2i, txt2img, img2txt):

    #      # Images->Text
    #     ranks = np.zeros(scores_i2t.shape[0])
    #     for index, score in enumerate(scores_i2t):
    #         inds = np.argsort(score)[::-1]
    #         # Score
    #         rank = 1e20
    #         for i in img2txt[index]:
    #             tmp = np.where(inds == i)[0][0]
    #             if tmp < rank:
    #                 rank = tmp
    #         ranks[index] = rank

    #     # Compute metrics
    #     tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    #     tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    #     tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    #     # Text->Images
    #     ranks = np.zeros(scores_t2i.shape[0])

    #     for index, score in enumerate(scores_t2i):
    #         inds = np.argsort(score)[::-1]
    #         tmp = txt2img[index]
    #         tmp2 = np.where(inds == txt2img[index])

    #         ranks[index] = np.where(inds == txt2img[index])[0][0]

    #     # Compute metrics
    #     ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    #     ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    #     ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    #     tr_mean = (tr1 + tr5 + tr10) / 3
    #     ir_mean = (ir1 + ir5 + ir10) / 3
    #     r_mean = (tr_mean + ir_mean) / 2

    #     agg_metrics = (tr1 + tr5 + tr10) / 3

    #     eval_result = {
    #         "txt_r1": tr1,
    #         "txt_r5": tr5,
    #         "txt_r10": tr10,
    #         "txt_r_mean": tr_mean,
    #         "img_r1": ir1,
    #         "img_r5": ir5,
    #         "img_r10": ir10,
    #         "img_r_mean": ir_mean,
    #         "r_mean": r_mean,
    #         "agg_metrics": agg_metrics,
    #     }
    #     return eval_result

