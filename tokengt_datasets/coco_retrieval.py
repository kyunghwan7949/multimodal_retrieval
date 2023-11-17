import json
from collections import defaultdict
import os
import os.path as op
from torch.utils.data import Dataset
from PIL import Image
from .bases import *
import torchvision.transforms as T
import torch
import torch.utils.data
import numpy as np
import pickle
import networkx as nx
import copy
import time
from utils.graph_tokenizer import SimpleTokenizer


class RetrievalGraphDataset(Dataset):

    def __init__(self, args, split, mlm=False, max_length=77, cache=False):
        super(RetrievalGraphDataset, self).__init__()
        
        self.args = args
        self.split = split
        self.cache = cache
        self.dataset_dir = op.join(self.args.root_dir, self.args.dataset_name)

        # self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        # self.img_path = op.join(self.dataset_dir, 'images')
        # # self.graph_path = f'/data/COCO/text_graph/{split}'
        # self.graph_path = f'/mount/IRRA/COCO/dp_graph_{split}'
        # self.cache_path = f'/mount/IRRA/COCO/cache'
        
        # # COCO 
        self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        self.img_path = op.join(self.dataset_dir, 'images')
        self.graph_path = f'/data/data2/IRRA/COCO/dp_graph_{split}'
        self.cache_path = f'/data/data2/IRRA/COCO/cache'
        self.annotation  = json.load(open(self.anno_path, "r"))
        self._add_instance_ids()
        
        # # RSTPReid
        # self.img_path = op.join('/data/data2/khahn/MGM_train/data/RSTPReid/imgs')
        # self.graph_path = f'/data/data2/khahn/MGM_train/data/RSTPReid/dp_graph/train'
        # self.cache_path = f'/data/data2/khahn/MGM_train/data/dp_graph/cache'
        # # # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/train.json", "r"))
        # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/splited_updated_train.json", "r"))

        self.mlm = mlm
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length

        

        
    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def _load_image(self, path):
        return Image.open(os.path.join(self.img_path, path)).convert("RGB")
    
    def _load_graph(self, pid):
        if self.cache:
            path = f'{self.split}_cache_{pid}.pkl'
            with open(op.join(self.cache_path, path),'rb') as f:
                data = pickle.load(f)
            return data
        else:
            path = f'{self.split}_graph_{pid}.pt'
            return torch.load(op.join(self.graph_path, path))

        
    def __getitem__(self, index):
        
        # COCO
        caption, image_path, image_id, pid = self.annotation[index].values()
        pid = torch.tensor(int(pid))

        graph = self._load_graph(pid)

   
        node_input_ids = self.get_node_info(graph['node_tokens'])

        edge_tokens = []
        for token in graph['edge_tokens']:
            edge_tokens.append(token)
            edge_tokens.append(token)
        edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
        edge_index = graph['edge_index']


        # 노드 마스킹
        if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
               edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
               edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]
            
        # 엣지 마스킹 : 엣지가 짝수여야하는데, edge가 max_length로 홀수로 짤릴수 있기 때문에 -1 추가
        if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
            elength = self.max_length - node_input_ids.size(0) 
            if elength % 2 != 0 :
                elength-=1
            edge_input_ids = edge_input_ids[:elength]
            edge_index = edge_index[:, :elength]


        lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

        caption_ids = torch.zeros(self.max_length,dtype=torch.long)
        caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
        mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)


        return {
            "pids": pid,
            "image_ids": image_id,
            "caption_ids": caption_ids,
            "node_data": node_input_ids,
            "edge_data": edge_input_ids,
            "edge_index": edge_index,
            "lap_eigvec": lap_eigvec.half(),
            "lap_eigval": lap_eigval.half(),
            "mlm_node": mlm_node,
            "mlm_edge": mlm_edge,
            "mlm_label" : mlm_label,
        }
        
        # # # RSTPReid
        # tmp = self.annotation[index]
        # pid, img_path, caption, split = self.annotation[index].values()
        
        # print("번호",pid)
        # pid = torch.tensor(int(pid))
        # graph = self._load_graph(pid)
             
       
        # '''
        # 조건1.  node_input_ids.size(0) + edge_input_ids.size(0) 가 self.max_length인 [77]보다 작거나 같아야한다.
        # 조건2. self._build_random_masked_tokens_and_labels에 인자로 입력될때는 최대 마스킹되는 토큰수가 self.max_length인 [77]보다 작거나 같아야한다.
        # '''

        # node_input_ids = self.get_node_info(graph['node_tokens'])
        
        # edge_tokens = []
        # for token in graph['edge_tokens']:
        #     edge_tokens.append(token)
        #     edge_tokens.append(token)
        # try:
        #     edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
        # except:
        #     print()
        # edge_index = graph['edge_index']

        
        # # # 노드 마스킹
        # # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
        # #        edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
        # #        edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]
        # total_length = node_input_ids.size(0) + edge_input_ids.size(0)
        # if total_length > self.max_length: # node+edge token수가 max_length 이하로 cut
        #     if node_input_ids.size(0) > self.max_length:  # # node token수가 max_length 이하로 cut
        #         node_input_ids = node_input_ids[:self.max_length]
        #         edge_input_ids = edge_input_ids[0:0] # 엣지 -> 0 
        #         edge_index = edge_index[:, 0:0] # 엣지 -> 0 
        #     else:
        #         edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
        #         edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]
            
        # # # 엣지 마스킹 : 엣지가 짝수여야하는데, edge가 max_length로 홀수로 짤릴수 있기 때문에 -1 추가
        # # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
        # #     elength = self.max_length - node_input_ids.size(0) 
        # #     if elength % 2 != 0 :
        # #         elength-=1
        # #     edge_input_ids = edge_input_ids[:elength]
        # #     edge_index = edge_index[:, :elength]
        

        # lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

        # caption_ids = torch.zeros(self.max_length, dtype=torch.long)
        # caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
        
        # # 노드 마스킹
        # mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids  , edge_input_ids, edge_index.t(), self.tokenizer)       
        
        # # 엣지 마스킹
        # # mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels_edge(node_input_ids  , edge_input_ids, edge_index.t(), self.tokenizer)       

        # return {
        #     "pids": pid,
        #     # "image_ids": image_id,
        #     "caption_ids": caption_ids,
        #     "node_data": node_input_ids,
        #     "edge_data": edge_input_ids,
        #     "edge_index": edge_index,
        #     "lap_eigvec": lap_eigvec.half(),
        #     "lap_eigval": lap_eigval.half(),
        #     "mlm_node": mlm_node,
        #     "mlm_edge": mlm_edge,
        #     "mlm_label" : mlm_label,

            
        # }
    
    
    def __len__(self) -> int:
        return len(self.annotation)
    
    def get_node_info(self, node_tokens):
        node_input_ids = []
        node_input_ids_orig = []
        for ntoken in node_tokens:
            input_ids = self.tokenizer.encode(ntoken)
            if len(input_ids)==0:
                input_ids = [0]
            node_input_ids+=input_ids

        return torch.tensor(node_input_ids)

    # # 노드 마스킹
    def _build_random_masked_tokens_and_labels(self, node_input_ids, edge_input_ids, edge_index, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 1 ~ 49405
        # print(mask)
        
        nids = node_input_ids.clone()
        eids = edge_input_ids.clone()
        
        node_labels = []
        edge_labels = torch.zeros(len(eids), dtype=torch.long)
        
        for i, token in enumerate(node_input_ids):
            if 3 < token < 26963:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        nids[i] = mask
                        
                        for eidx, edge in enumerate(edge_index):
                            if edge[0] == i or edge[1] == i:
                                edge_labels[eidx] = eids[eidx]
                                eids[eidx]=mask 

                                # edge_labels[eidx] = mask
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        nids[i] = random.choice(token_range)
                        for eidx, edge in enumerate(edge_index):
                            if edge[0] == i or edge[1] == i:
                                edge_labels[eidx] = eids[eidx]
                                eids[eidx] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    node_labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    node_labels.append(torch.tensor(0))
            else:
                node_labels.append(torch.tensor(0))
        
        if all(l == 0 for l in node_labels):
            # at least mask 1
            node_labels[0] = nids[0]
            nids[0] = mask
            for eidx, edge in enumerate(edge_index):
                if edge[0] == 0 or edge[1] == 0:
                    edge_labels[eidx] = eids[eidx]
                    eids[eidx] = mask
                    
        
        padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        
        labels = torch.tensor(node_labels + edge_labels.tolist())

        padded_labels[:len(labels)] = labels
        
        return nids, eids, padded_labels


    ## 엣지 마스킹
    def _build_random_masked_tokens_and_labels_edge(self, node_input_ids, edge_input_ids, edge_index, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 1 ~ 49405

        nids = node_input_ids.clone()
        eids = edge_input_ids.clone()
        
        node_labels = torch.zeros(len(nids), dtype=torch.long)
        edge_labels = []
        
        for i, token in enumerate(edge_input_ids[::2]):
            if 3 < token < 26963:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        eids[i*2] = mask
                        eids[i*2+1] = mask

                        target_nodes = edge_index[i*2]
                        node_labels[target_nodes[0]] = node_input_ids[target_nodes[0]]
                        node_labels[target_nodes[1]] = node_input_ids[target_nodes[1]]

                        nids[target_nodes[0]]=mask
                        nids[target_nodes[1]]=mask 
                        
                        # print("마스킹 엣지:", (i*2, i*2+1))
                        # print("마스킹 노드:", target_nodes[0], target_nodes[1])

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        randn = random.choice(token_range)
                        eids[i*2] = randn
                        eids[i*2+1] = randn

                        target_nodes = edge_index[i*2]
                        node_labels[target_nodes[0]] = node_input_ids[target_nodes[0]]
                        node_labels[target_nodes[1]] = node_input_ids[target_nodes[1]]

                        nids[target_nodes[0]]=randn
                        nids[target_nodes[1]]=randn 
                        
                        # print("마스킹 엣지:", (i*2, i*2+1))
                        # print("마스킹 노드:", target_nodes[0], target_nodes[1])

                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    edge_labels.append(int(token))
                    edge_labels.append(int(token))

                else:
                    # no masking token (will be ignored by loss function later)
                    edge_labels.append(0)
                    edge_labels.append(0)

            else:
                edge_labels.append(0)
                edge_labels.append(0)

                   
        
        padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        
        labels = torch.tensor(node_labels.tolist() + edge_labels)

        padded_labels[:len(labels)] = labels
        
        return nids, eids, padded_labels

    
    # def _build_random_masked_tokens_and_labels(self, node_input_ids, edge_input_ids, edge_index, tokenizer):
    #     """
    #     Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    #     :param tokens: list of int, tokenized sentence.
    #     :return: (list of int, list of int), masked tokens and related labels for MLM prediction
    #     """
    #     mask = tokenizer.encoder["<mask>"]
    #     token_range = list(range(4, len(tokenizer.encoder)-1)) # 1 ~ 49405
        
    #     nids = node_input_ids.clone()
    #     eids = edge_input_ids.clone()
        
    #     node_labels = []
    #     edge_labels = torch.zeros(len(eids), dtype=torch.long)
        
    #     for i, token in enumerate(node_input_ids):
    #         if 3 < token < 26963:
    #             prob = random.random()
    #             # mask token with 15% probability
    #             if prob < 0.15:
    #                 prob /= 0.15

    #                 # 80% randomly change token to mask token
    #                 if prob < 0.8:
    #                     nids[i] = mask
                        
    #                     for eidx, edge in enumerate(edge_index):
    #                         if edge[0] == i or edge[1] == i:
    #                             edge_labels[eidx] = eids[eidx]
    #                             eids[eidx]=mask 

    #                             # edge_labels[eidx] = mask
    #                 # 10% randomly change token to random token
    #                 elif prob < 0.9:
    #                     nids[i] = random.choice(token_range)
    #                     for eidx, edge in enumerate(edge_index):
    #                         if edge[0] == i or edge[1] == i:
    #                             edge_labels[eidx] = eids[eidx]
    #                             eids[eidx] = random.choice(token_range)

    #                 # -> rest 10% randomly keep current token

    #                 # append current token to output (we will predict these later)
    #                 node_labels.append(token)
    #             else:
    #                 # no masking token (will be ignored by loss function later)
    #                 node_labels.append(torch.tensor(0))
    #         else:
    #             node_labels.append(torch.tensor(0))
        
    #     if all(l == 0 for l in node_labels):
    #         # at least mask 1
    #         node_labels[0] = nids[0]
    #         nids[0] = mask
    #         for eidx, edge in enumerate(edge_index):
    #             if edge[0] == 0 or edge[1] == 0:
    #                 edge_labels[eidx] = eids[eidx]
    #                 eids[eidx] = mask
                    
        
    #     padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        
    #     labels = torch.tensor(node_labels + edge_labels.tolist())

    #     padded_labels[:len(labels)] = labels
        
    #     return nids, eids, padded_labels


    def preprocess_item(self, n_node, edge_index):

        N = n_node
        dense_adj = torch.zeros([N, N], dtype=torch.bool)
        dense_adj[edge_index[0, :], edge_index[1, :]] = True
        in_degree = dense_adj.long().sum(dim=1).view(-1) # number of degree == number of node
        
        lap_eigvec, lap_eigval = lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]

        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec) # [N, N]

        return lap_eigvec, lap_eigval 


    def eig(self, sym_mat):
        # (sorted) eigenvectors with numpy
        EigVal, EigVec = np.linalg.eigh(sym_mat)

        # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
        eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
        eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float()  # [N (channels),]
        return eigvec, eigval  # [N, N (channels)]  [N (channels),]


    def lap_eig(self, dense_adj, number_of_nodes, in_degree):
        """
        Graph positional encoding v/ Laplacian eigenvectors
        https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
        """
        dense_adj = dense_adj.detach().float().numpy()
        in_degree = in_degree.detach().float().numpy()

        # Laplacian
        A = dense_adj
        N = np.diag(in_degree.clip(1) ** -0.5)
        L = np.eye(number_of_nodes) - N @ A @ N

        eigvec, eigval = self.eig(L)
        return eigvec, eigval  # [N, N (channels)]  [N (channels),]


class RetrievalGraphEvalDataset(Dataset):

    def __init__(self, args, split, max_length=77):
        super(RetrievalGraphEvalDataset, self).__init__()
        
        self.args = args
        self.split = split
        self.dataset_dir = op.join(self.args.root_dir, self.args.dataset_name)
        # self.transform = self.transform
        
        
        # COCO 
        self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        self.img_path = op.join(self.dataset_dir, 'images')
        self.graph_path = f'/data/data2/IRRA/COCO/dp_graph_{split}'
        self.cache_path = f'/data/data2/IRRA/COCO/cache'
        self.annotation  = json.load(open(self.anno_path, "r"))
        
        # ## RSTPReid
        # self.img_path = op.join('/data/data2/khahn/MGM_train/data/RSTPReid/imgs')
        # self.graph_path = f'/data/data2/khahn/MGM_train/data/RSTPReid/dp_graph/val'
        # self.cache_path = f'/data/data2/khahn/MGM_train/data/dp_graph/cache'
        # # # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/val.json", "r"))
        # # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/splited_updated_val.json", "r"))
        # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/val.json", "r"))
        
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length
        self._add_instance_ids()

        self.text = []
        self.graph = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
    
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            
            # COO
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                  
                graph_dict = {}
                graph = self._load_graph(txt_id)
                node_input_ids = self.get_node_info(graph['node_tokens'])
                
                edge_tokens = []
                for token in graph['edge_tokens']:
                    edge_tokens.append(token)
                    edge_tokens.append(token)
                edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
                edge_index = graph['edge_index']

                # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
                #     edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
                #     edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]

                # 엣지가 짝수여야하는데, edge가 max_length로 홀수로 짤릴수 있기 때문에 -1 추가
                if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
                    elength = self.max_length - node_input_ids.size(0) 
                    if elength % 2 != 0 :
                        elength-=1
                    edge_input_ids = edge_input_ids[:elength]
                    edge_index = edge_index[:, :elength]

                lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

                caption_ids = torch.zeros(self.max_length, dtype=torch.long)
                caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
                mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)

                graph_dict["node_data"] = node_input_ids
                graph_dict["edge_data"] = edge_input_ids
                graph_dict["caption_ids"] = caption_ids
                graph_dict["edge_index"] = edge_index
                graph_dict["lap_eigvec"] = lap_eigvec.half()
                graph_dict["lap_eigval"] = lap_eigval.half()
                graph_dict["mlm_node"] = mlm_node
                graph_dict["mlm_edge"] = mlm_edge
                graph_dict["mlm_label"] = mlm_label
                            
                self.graph.append(graph_dict)
                txt_id += 1
            
            # # RSTPReid
            # self.image.append(ann["img_path"])
            # self.img2txt[img_id] = []
            
            # for i, caption in enumerate(ann["captions"]):
                  
            #     graph_dict = {}
            #     graph = self._load_graph(txt_id)
            #     # graph = self._load_graph(img_id)
                
            #     node_input_ids = self.get_node_info(graph['node_tokens'])
                
            #     edge_tokens = []
            #     for token in graph['edge_tokens']:
            #         edge_tokens.append(token)
            #         edge_tokens.append(token)
            #     edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
            #     edge_index = graph['edge_index']

            #     # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
            #     #     edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
            #     #     edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]
                
            #     # # 노드 마스킹
            #     # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
            #     #     edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
            #     #     edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]
            #     total_length = node_input_ids.size(0) + edge_input_ids.size(0)
            #     if total_length > self.max_length: # node+edge token수가 max_length 이하로 cut
            #         if node_input_ids.size(0) > self.max_length:  # # node token수가 max_length 이하로 cut
            #             node_input_ids = node_input_ids[:self.max_length]
            #             edge_input_ids = edge_input_ids[0:0] # 엣지 -> 0 
            #             edge_index = edge_index[:, 0:0] # 엣지 -> 0 
            #         else:
            #             edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
            #             edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]
                    
            #     # # 엣지 마스킹 : 엣지가 짝수여야하는데, edge가 max_length로 홀수로 짤릴수 있기 때문에 -1 추가
            #     # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
            #     #     elength = self.max_length - node_input_ids.size(0) 
            #     #     if elength % 2 != 0 :
            #     #         elength-=1
            #     #     edge_input_ids = edge_input_ids[:elength]
            #     #     edge_index = edge_index[:, :elength]

            #     lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

            #     caption_ids = torch.zeros(self.max_length, dtype=torch.long)
            #     caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
                
            #     # # 노드 마스킹
            #     mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)
                
            #     # # 엣지 마스킹 
            #     # mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels_edge(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)

            #     graph_dict["node_data"] = node_input_ids
            #     graph_dict["edge_data"] = edge_input_ids
            #     graph_dict["caption_ids"] = caption_ids
            #     graph_dict["edge_index"] = edge_index
            #     graph_dict["lap_eigvec"] = lap_eigvec.half()
            #     graph_dict["lap_eigval"] = lap_eigval.half()
            #     graph_dict["mlm_node"] = mlm_node
            #     graph_dict["mlm_edge"] = mlm_edge
            #     graph_dict["mlm_label"] = mlm_label
                            
            #     self.graph.append(graph_dict)
            #     txt_id += 1
            
        

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def _load_image(self, path):
        return Image.open(os.path.join(self.img_path, path)).convert("RGB")

    def _load_graph(self, pid):
        path = f'{self.split}_graph_{pid}.pt'
        return torch.load(op.join(self.graph_path, path))
    
    def __getitem__(self, index):

        image_path = os.path.join(self.img_path, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.transforms(image)

        return {"image": image, "index": index}
    
    def __len__(self) -> int:
        return len(self.annotation)
    
    def get_node_info(self, node_tokens):
        node_input_ids = []
        node_input_ids_orig = []
        for ntoken in node_tokens:
            input_ids = self.tokenizer.encode(ntoken)
            if len(input_ids)==0:
                input_ids = [0]
            else:
                node_input_ids_orig.append(input_ids)
            node_input_ids+=input_ids

        return torch.tensor(node_input_ids)

    def get_edge_info(self, node_input_ids_orig, n_node, edge_tokens, edge_index):
        # node_input_ids_orig[-1] = [192,184, 543]
        # node_input_ids_orig[8] = [192,184, 543]

        # 엣지는 양방향이라서 엣지 토큰도 두배로 만들어줌
        orig_tokens = []
        for token in edge_tokens:
            orig_tokens.append(token)
            orig_tokens.append(token)
        tmp_tokens = copy.deepcopy(orig_tokens)

        # 노드를 토크나이저했을 떄 하나 이상의 토큰이 나올 수 있기 때문에 각 토큰을 노드로 봄
        orig_index = edge_index.clone().tolist()
        tmp_index = edge_index.clone().tolist()
        
        target = []
        orig_target = [idx for idx, nid in enumerate(node_input_ids_orig) if len(nid) != 1]
        
        extra = 0
        for i, nid in enumerate(node_input_ids_orig[:-1]):
            if len(nid)==1: continue
            for _ in range(len(nid)-1): 
                for j, edge in enumerate(orig_index):                       
                        if edge[0] > i:
                            tmp_index[j][0]+=1
                        if edge[1] > i:
                            tmp_index[j][1]+=1
            target.append(i+extra)
            extra+=(len(nid)-1)
        
        if len(node_input_ids_orig[-1]) != 1:
            target.append(len(node_input_ids_orig) -1 + extra)

        new_index = copy.deepcopy(tmp_index)
        new_tokens = copy.deepcopy(tmp_tokens)

        for ot, t in zip(orig_target, target):
            node_ids = node_input_ids_orig[ot]
            for idx in tmp_index:
                if idx[0] == t or idx[1] == t:
                    for i in range(1,len(node_ids)):
                        if idx[0] == t:
                            new_index.append([idx[0]+i,idx[1]])
                        elif idx[1] == t:
                            new_index.append([idx[0],idx[1]+i])
                        new_tokens.append(tmp_tokens[i])

        edge_input_ids = []
        output_index = []
        for token, idx in zip(new_tokens, new_index):
            input_ids = self.tokenizer.encode(token)
            if len(input_ids) != 1:
                for j in range(len(input_ids)):
                    output_index.append(idx)                
            else:
                output_index.append(idx)    
            edge_input_ids += input_ids


        edge_input_ids = torch.tensor(edge_input_ids)
        output_index =  torch.tensor(output_index).t()
        
        if n_node + len(edge_input_ids) > self.max_length:
            edge_input_ids = edge_input_ids[:self.max_length-n_node]
            output_index = output_index[:,:self.max_length-n_node]

        return edge_input_ids, output_index

    # # 노드 마스킹
    def _build_random_masked_tokens_and_labels(self, node_input_ids, edge_input_ids, edge_index, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 1 ~ 49405
        
        nids = node_input_ids.clone()
        eids = edge_input_ids.clone()
        
        node_labels = []
        edge_labels = torch.zeros(len(eids), dtype=torch.long)
        
        for i, token in enumerate(node_input_ids):
            if 3 < token < 26963:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        nids[i] = mask
                        
                        for eidx, edge in enumerate(edge_index):
                            if edge[0] == i or edge[1] == i:
                                edge_labels[eidx] = eids[eidx]
                                eids[eidx]=mask 

                                # edge_labels[eidx] = mask
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        nids[i] = random.choice(token_range)
                        for eidx, edge in enumerate(edge_index):
                            if edge[0] == i or edge[1] == i:
                                edge_labels[eidx] = eids[eidx]
                                eids[eidx] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    node_labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    node_labels.append(torch.tensor(0))
            else:
                node_labels.append(torch.tensor(0))
        
        if all(l == 0 for l in node_labels):
            # at least mask 1
            node_labels[0] = nids[0]
            nids[0] = mask
            for eidx, edge in enumerate(edge_index):
                if edge[0] == 0 or edge[1] == 0:
                    edge_labels[eidx] = eids[eidx]
                    eids[eidx] = mask
                    
        
        padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        
        labels = torch.tensor(node_labels + edge_labels.tolist())

        padded_labels[:len(labels)] = labels
        
        return nids, eids, padded_labels


    ## 엣지 마스킹
    def _build_random_masked_tokens_and_labels_edge(self, node_input_ids, edge_input_ids, edge_index, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 1 ~ 49405

        nids = node_input_ids.clone()
        eids = edge_input_ids.clone()
        
        node_labels = torch.zeros(len(nids), dtype=torch.long)
        edge_labels = []
        
        for i, token in enumerate(edge_input_ids[::2]):
            if 3 < token < 26963:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        eids[i*2] = mask
                        eids[i*2+1] = mask

                        target_nodes = edge_index[i*2]
                        node_labels[target_nodes[0]] = node_input_ids[target_nodes[0]]
                        node_labels[target_nodes[1]] = node_input_ids[target_nodes[1]]

                        nids[target_nodes[0]]=mask
                        nids[target_nodes[1]]=mask 
                        
                        # print("마스킹 엣지:", (i*2, i*2+1))
                        # print("마스킹 노드:", target_nodes[0], target_nodes[1])

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        randn = random.choice(token_range)
                        eids[i*2] = randn
                        eids[i*2+1] = randn

                        target_nodes = edge_index[i*2]
                        node_labels[target_nodes[0]] = node_input_ids[target_nodes[0]]
                        node_labels[target_nodes[1]] = node_input_ids[target_nodes[1]]

                        nids[target_nodes[0]]=randn
                        nids[target_nodes[1]]=randn 
                        
                        # print("마스킹 엣지:", (i*2, i*2+1))
                        # print("마스킹 노드:", target_nodes[0], target_nodes[1])

                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    edge_labels.append(int(token))
                    edge_labels.append(int(token))

                else:
                    # no masking token (will be ignored by loss function later)
                    edge_labels.append(0)
                    edge_labels.append(0)

            else:
                edge_labels.append(0)
                edge_labels.append(0)   
        
        padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        
        labels = torch.tensor(node_labels.tolist() + edge_labels)

        padded_labels[:len(labels)] = labels
        
        return nids, eids, padded_labels
