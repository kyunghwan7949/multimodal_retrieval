from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import time
import datetime
from tqdm import tqdm


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


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
        
        # model.float()
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_features = []

        for i in range(0, num_text, text_bs):

            text = texts[i : min(num_text, i + text_bs)]
            text_input = torch.stack(text).to(device)
            text_feat = model.encode_text(text_input).float()
            text_feat = F.normalize(text_feat, dim=-1)
            text_features.append(text_feat)

        text_features = torch.cat(text_features, dim=0)


        image_features = []
        for samples in tqdm(data_loader):
            image = samples["image"]

            image = image.to(device)
            image_feat = model.encode_image(image).float()
            image_feat = F.normalize(image_feat, dim=-1)
            image_features.append(image_feat)

        image_features = torch.cat(image_features, dim=0)
        
        sims_matrix_i2t = image_features @ text_features.t()
        sims_matrix_t2i = sims_matrix_i2t.t()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))
        
        return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy()

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

