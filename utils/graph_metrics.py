from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import time
import datetime
from tqdm import tqdm



class Evaluator():
    def __init__(self, data_loader):
        # self.img_loader = img_loader # gallery
        # self.txt_loader = txt_loader # query
        self.data_loader = data_loader
        self.logger = logging.getLogger("IRRA.eval")

    def eval(self, model):
        score_i2t, score_t2i = self.compute_sim_matrix(model, self.data_loader)

        eval_result = self.report_metrics(
            score_i2t,
            score_t2i,
            self.data_loader.dataset.txt2img,
            self.data_loader.dataset.img2txt
        )
        print(eval_result)
        return eval_result
    

    
    @torch.no_grad()
    def compute_sim_matrix(self, model, data_loader):
        
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        logging.info("Computing features for evaluation...")
        start_time = time.time()

        graphs = data_loader.dataset.graph
        num_graph = len(graphs)
        graph_bs = 256
        graph_features = []

        for i in range(0, num_graph, graph_bs):

            graph = graphs[i : min(num_graph, i + graph_bs)]
            batch_graph = self.batch_collate(graph)
            batch_graph = {k: v.to(device) for k, v in batch_graph.items()}

            graph_feat = model.tokengt.encoder(batch_graph, mlm=False)
            g_feats = model.tokengt_proj(graph_feat[:, 0, :]).float()

            g_feats = g_feats / g_feats.norm(dim=1, keepdim=True)

            graph_features.append(g_feats)

        graph_features = torch.cat(graph_features, dim=0)

        image_features = []
        for samples in tqdm(data_loader):
            image = samples["image"].half()

            image = image.to(device)
            image_feat = model.base_model(image)
            i_feats = image_feat[:, 0, :].float()

            i_feats = F.normalize(i_feats, dim=-1)
            image_features.append(i_feats)

        image_features = torch.cat(image_features, dim=0)

        sims_matrix_i2t = image_features @ graph_features.t()
        sims_matrix_t2i = sims_matrix_i2t.t()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy()
    

    def batch_collate(self, batch):
        
        node_num = torch.tensor([i['node_data'].size(0) for i in batch])
        edge_num = torch.tensor([i['edge_index'].size(1) for i in batch])
        max_n = max(node_num)

        edge_index = [i['edge_index'] for i in batch] # [2, sum(edge_num)]
        edge_index = torch.cat(edge_index, dim=1)

        # edge_data = [i['edge_data'] for i in batch] # [sum(edge_num), De]
        # edge_data = torch.cat(edge_data)

        node_data = [i['node_data'] for i in batch] # [sum(node_num), Dn]
        node_data = torch.cat(node_data)

        # [sum(node_num), Dl] = [sum(node_num), max_n]
        lap_eigvec = [i['lap_eigvec'] for i in batch]
        lap_eigvec = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigvec])

        lap_eigval = [i['lap_eigval'] for i in batch]
        lap_eigval = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigval])

        item = dict(
            edge_index=edge_index,
            # edge_data=edge_data,
            node_data=node_data,
            lap_eigvec=lap_eigvec,
            lap_eigval=lap_eigval,
            node_num=node_num,
            edge_num=edge_num,
        )
        
        return item

    @torch.no_grad()
    def report_metrics(self, scores_i2t, scores_t2i, txt2img, img2txt):

         # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            tmp = txt2img[index]
            tmp2 = np.where(inds == txt2img[index])

            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        return eval_result

