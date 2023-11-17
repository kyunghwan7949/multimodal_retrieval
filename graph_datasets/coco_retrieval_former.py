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
from utils.tokengt_tokenizer import SimpleTokenizer


class RetrievalGraphDataset(Dataset):

    def __init__(self, args, split, transforms, mlm=False, max_length=77, cache=False):
        super(RetrievalGraphDataset, self).__init__()
        
        self.args = args
        self.split = split
        self.cache = cache
        self.dataset_dir = op.join(self.args.root_dir, self.args.dataset_name)
        
        # COCO 
        self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        self.img_path = op.join(self.dataset_dir, 'images')
        self.graph_path = f'/data/data2/IRRA/COCO/dp_graph_{split}'
        self.cache_path = f'/data/data2/IRRA/COCO/cache'
        self.annotation  = json.load(open(self.anno_path, "r"))
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        # # RSTPReid
        # self.img_path = op.join('/data/data2/khahn/MGM_train/data/RSTPReid/imgs')
        # self.graph_path = f'/data/data2/khahn/MGM_train/data/dp_graph/dp_graph_{split}_splited_update'
        # self.cache_path = f'/data/data2/khahn/MGM_train/data/dp_graph/cache'
        # # # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/train.json", "r"))
        # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/splited_updated_train.json", "r"))
        
        # self.mlm = mlm
        # self.transforms = transforms
        # self.tokenizer = SimpleTokenizer()
        # self.max_length = max_length

        # self._add_instance_ids()
        
        # self.img_ids = {}
        # n = 0
        # for ann in self.annotation:
        #     img_id = ann["img_path"]
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1
        # t=3
        
        

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
        
        ## COCO
        caption, image_path, image_id, pid = self.annotation[index].values()
        image = self._load_image(image_path)
        pid = torch.tensor(int(pid))

        if self.transforms is not None:
            image = self.transforms(image)

        graph = self._load_graph(pid)
     
        node_input_ids = self.get_node_info(graph['node_tokens'])

        edge_tokens = []
        for token in graph['edge_tokens']:
            edge_tokens.append(token)
            edge_tokens.append(token)
        edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
        edge_index = graph['edge_index']

        if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
            edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
            edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]

        lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

        caption_ids = torch.zeros(self.max_length,dtype=torch.long)
        caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
        # mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)
        mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels_edge_node(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)

        return {
            "pids": pid,
            "image_ids": image_id,
            "images": image,
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
        # # print(tmp)
        # id, image_path, caption, split, pid = self.annotation[index].values()
        # pid = torch.tensor(int(pid))
        # graph = self._load_graph(pid) 
             
        # image = self._load_image(image_path)
        # if self.transforms is not None:
        #     image = self.transforms(image)
        # '''
        # 조건1.  node_input_ids.size(0) + edge_input_ids.size(0) 가 self.max_length인 [77]보다 작거나 같아야한다.
        # 조건2. self._build_random_masked_tokens_and_labels에 인자로 입력될때는 최대 마스킹되는 토큰수가 self.max_length인 [77]보다 작거나 같아야한다.
        # '''

        # node_input_ids = self.get_node_info(graph['node_tokens'])
        
        # edge_tokens = []
        # for token in graph['edge_tokens']:
        #     edge_tokens.append(token)
        #     edge_tokens.append(token)
        # edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
        # edge_index = graph['edge_index']

        
        # # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
        # #        edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
        # #        edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]
            
        # # 조건 1에 대한 처리
        # total_length = node_input_ids.size(0) + edge_input_ids.size(0)
        # if total_length > self.max_length: # node+edge token수가 max_length 이하로 cut
        #     if node_input_ids.size(0) > self.max_length:  # # node token수가 max_length 이하로 cut
        #         node_input_ids = node_input_ids[:self.max_length]
        #         edge_input_ids = edge_input_ids[0:0] # 엣지 -> 0 
        #         edge_index = edge_index[:, 0:0] # 엣지 -> 0 
        #     else:
        #         edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
        #         edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]

        # lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

        # caption_ids = torch.zeros(self.max_length, dtype=torch.long)
        # caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
        
        
        # # node_edge_size =  torch.cat([node_input_ids, edge_input_ids])
        # # caption_size = caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)]

        # mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids  , edge_input_ids, edge_index.t(), self.tokenizer)       

        # return {
        #     "pids": pid,
        #     # "image_ids": image_id,
        #     "images": image,
        #     "caption_ids": caption_ids,
        #     "node_data": node_input_ids,
        #     "edge_data": edge_input_ids,
        #     "edge_index": edge_index,
        #     "lap_eigvec": lap_eigvec.half(),
        #     "lap_eigval": lap_eigval.half(),
        #     "mlm_node": mlm_node,
        #     "mlm_edge": mlm_edge,
        #     "mlm_label" : mlm_label,
        #     # "node_info" : node_input_ids.size(0),
        #     # "edge_info" : edge_input_ids.size(0),
        #     # "node_edge_size" : node_edge_size.size(0),
        #     # "caption_size" : caption_size.size(0),
            
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

    ## 엣지 -> 노드 순 마스킹
    '''
    마스킹 조건
    조건1. 내가 사용하는 그래프는 양방향 그래프이고, (노드A, 노드B)엣지와 (노드A, 노드B)엣지는 노드A와 노드B 연결된 엣지들이다.
    조건2. (노드A, 노드B)엣지가 마스킹 되었다면 무조건 (노드B, 노드A)엣지는 마스킹 되어야 한다.
    조건2. (노드A, 노드B) 엣지에서 엣지 마스킹 진행 후, 노드A와 노드B를 마스킹하게 되면, (노드B, 노드A) 엣지에서 엣지 마스킹 진행 후, 노드A와  노드B는 이미 앞서 노드 마스킹 하였으므로 하지 않는다.
    조건3. 마스킹된 엣지와 연결된 노드들은 모두 마스킹한다.
    '''
    def _build_random_masked_tokens_and_labels_edge_node(self, node_input_ids, edge_input_ids, edge_index, tokenizer):
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 4 ~ 26962 / 4 ~ 3084

        nids = node_input_ids.clone() # 노드 ID
        eids = edge_input_ids.clone() # 엣지 ID

        node_labels = torch.zeros(len(nids), dtype=torch.long) # 각 node 토큰 레이블을 0으로 초기화
        edge_labels = torch.zeros(len(eids), dtype=torch.long) # 각 edge 토큰 레이블을 0으로 초기화

        for i in range(0, len(eids)-1, 2):
            # edge_input_ids가 두 개씩 한 쌍으로 나타나므로, 2씩 증가시키면서 loop 수행
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% 확률로 두 엣지를 함께 마스킹
                if prob < 0.8:
                    edge_labels[i] = eids[i]
                    edge_labels[i+1] = eids[i+1]
                    eids[i] = mask
                    eids[i+1] = mask
                    connected_nodes = edge_index[i//2]  # connected_nodes는 [노드B, 노드D]를 가정
                    for node in connected_nodes:
                        node_labels[node] = nids[node]
                        nids[node] = mask
                        
                    # 마스킹 된 엣지와 그에 연결된 노드 출력
                    # print(f"마스킹 엣지: {i}, {i+1}")
                    # print(f"마스킹 노드: {connected_nodes.tolist()}")
                    

                # 10% 확률로 두 엣지를 함께 랜덤한 토큰으로 마스킹
                elif prob < 0.9:
                    random_token = random.choice(token_range)
                    edge_labels[i] = eids[i]
                    edge_labels[i+1] = eids[i+1]
                    eids[i] = random_token
                    eids[i+1] = random_token
                    connected_nodes = edge_index[i//2]
                    for node in connected_nodes:
                        node_labels[node] = nids[node]
                        nids[node] = random_token
                        
                    # 마스킹 된 엣지와 그에 연결된 노드 출력
                    # print(f"마스킹 엣지: {i}, {i+1}")
                    # print(f"마스킹 노드: {connected_nodes.tolist()}")    
                    # print("-------------------")
                    

        # print("node_input_ids :", node_input_ids)
        # print("nids :", nids)
        # print("edge_input_ids :", edge_input_ids)
        # print("eids :", eids)
        # print("---------")
        

        # 나머지 10%는 그대로 둡니다 (마스킹하지 않습니다)

        padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        labels = torch.tensor(node_labels.tolist() + edge_labels.tolist())
        padded_labels[:len(labels)] = labels



        # padded_labels = torch.zeros(self.max_length, dtype=torch.long)

        # labels = torch.cat((edge_labels, node_labels))

        # padded_labels[:len(labels)] = labels

        return nids, eids, padded_labels
    

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

    def __init__(self, args, split, transforms, max_length=77):
        super(RetrievalGraphEvalDataset, self).__init__()
        
        self.args = args
        self.split = split
        self.dataset_dir = op.join(self.args.root_dir, self.args.dataset_name)
        
        # COCO 
        self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        self.img_path = op.join(self.dataset_dir, 'images')
        self.graph_path = f'/data/data2/IRRA/COCO/dp_graph_{split}'
        self.cache_path = f'/data/data2/IRRA/COCO/cache'
        self.annotation  = json.load(open(self.anno_path, "r"))

        # # ## RSTPReid
        # self.img_path = op.join('/data/data2/khahn/MGM_train/data/RSTPReid/imgs')
        # self.graph_path = f'/data/data2/khahn/MGM_train/data/dp_graph/dp_graph_{split}_splited_update'
        # self.cache_path = f'/data/data2/khahn/MGM_train/data/dp_graph/cache'
        # # # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/val.json", "r"))
        # # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/splited_updated_val.json", "r"))
        # self.annotation  = json.load(open("/data/data2/khahn/MGM_train/data/RSTPReid/val.json", "r"))




        self.transforms = transforms

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
            
            ## COCO
            self.image.append(ann["img_path"])
            self.img2txt[img_id] = []
            
            for i, caption in enumerate(ann["captions"]):
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id

                graph_dict = {}
                graph = self._load_graph(txt_id)
                node_input_ids = self.get_node_info(graph['node_tokens'])
                
                edge_tokens = []
                for token in graph['edge_tokens']:
                    edge_tokens.append(token)
                    edge_tokens.append(token)
                edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
                edge_index = graph['edge_index']

                if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
                    edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
                    edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]

                lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

                caption_ids = torch.zeros(self.max_length, dtype=torch.long)
                caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
                # mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)
                mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels_edge_node(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)

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
            
            # ##  RSTPReid
            # self.image.append(ann["img_path"])
            # self.img2txt[img_id] = []
            
            # for i, caption in enumerate(ann["captions"]):
            #     self.img2txt[img_id].append(txt_id)
            #     self.txt2img[txt_id] = img_id
                  
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
                
            #     # 조건 1에 대한 처리
            #     total_length = node_input_ids.size(0) + edge_input_ids.size(0)
            #     if total_length > self.max_length:
            #         if node_input_ids.size(0) > self.max_length: 
            #             node_input_ids = node_input_ids[:self.max_length]
            #             edge_input_ids = edge_input_ids[0:0] # 빈 tensor
            #             edge_index = edge_index[:, 0:0] # 빈 tensor
            #         else:
            #             edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
            #             edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]

            #     lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

            #     caption_ids = torch.zeros(self.max_length, dtype=torch.long)
            #     caption_ids[:node_input_ids.size(0) + edge_input_ids.size(0)] = torch.cat([node_input_ids, edge_input_ids])
                
            #     # mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels_edge_node(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)
            #     mlm_node, mlm_edge, mlm_label = self._build_random_masked_tokens_and_labels(node_input_ids, edge_input_ids, edge_index.t(), self.tokenizer)

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
        ab = self.annotation[index]
        ## COCO
        # image_path = os.path.join(self.img_path, self.annotation[index]["image"])
        
        ## # RSTPReid
        image_path = os.path.join(self.img_path, self.annotation[index]["img_path"])
        
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
    
    
    ## 엣지 -> 노드 순 마스킹
    '''
    마스킹 조건
    조건1. 내가 사용하는 그래프는 양방향 그래프이고, (노드A, 노드B)엣지와 (노드A, 노드B)엣지는 노드A와 노드B 연결된 엣지들이다.
    조건2. (노드A, 노드B)엣지가 마스킹 되었다면 무조건 (노드B, 노드A)엣지는 마스킹 되어야 한다.
    조건2. (노드A, 노드B) 엣지에서 엣지 마스킹 진행 후, 노드A와 노드B를 마스킹하게 되면, (노드B, 노드A) 엣지에서 엣지 마스킹 진행 후, 노드A와  노드B는 이미 앞서 노드 마스킹 하였으므로 하지 않는다.
    조건3. 마스킹된 엣지와 연결된 노드들은 모두 마스킹한다.
    '''
    def _build_random_masked_tokens_and_labels_edge_node(self, node_input_ids, edge_input_ids, edge_index, tokenizer):
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 4 ~ 26962 / 4 ~ 3084

        nids = node_input_ids.clone() # 노드 ID
        eids = edge_input_ids.clone() # 엣지 ID

        node_labels = torch.zeros(len(nids), dtype=torch.long) # 각 node 토큰 레이블을 0으로 초기화
        edge_labels = torch.zeros(len(eids), dtype=torch.long) # 각 edge 토큰 레이블을 0으로 초기화

        for i in range(0, len(eids)-1, 2):
            # edge_input_ids가 두 개씩 한 쌍으로 나타나므로, 2씩 증가시키면서 loop 수행
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% 확률로 두 엣지를 함께 마스킹
                if prob < 0.8:
                    edge_labels[i] = eids[i]
                    edge_labels[i+1] = eids[i+1]
                    eids[i] = mask
                    eids[i+1] = mask
                    connected_nodes = edge_index[i//2]  # connected_nodes는 [노드B, 노드D]를 가정
                    for node in connected_nodes:
                        node_labels[node] = nids[node]
                        nids[node] = mask
                        
                    # 마스킹 된 엣지와 그에 연결된 노드 출력
                    # print(f"마스킹 엣지: {i}, {i+1}")
                    # print(f"마스킹 노드: {connected_nodes.tolist()}")
                    

                # 10% 확률로 두 엣지를 함께 랜덤한 토큰으로 마스킹
                elif prob < 0.9:
                    random_token = random.choice(token_range)
                    edge_labels[i] = eids[i]
                    edge_labels[i+1] = eids[i+1]
                    eids[i] = random_token
                    eids[i+1] = random_token
                    connected_nodes = edge_index[i//2]
                    for node in connected_nodes:
                        node_labels[node] = nids[node]
                        nids[node] = random_token
                        
                    # 마스킹 된 엣지와 그에 연결된 노드 출력
                    # print(f"마스킹 엣지: {i}, {i+1}")
                    # print(f"마스킹 노드: {connected_nodes.tolist()}")    
                    # print("-------------------")
                    

        # print("node_input_ids :", node_input_ids)
        # print("nids :", nids)
        # print("edge_input_ids :", edge_input_ids)
        # print("eids :", eids)
        # print("---------")
        

        # 나머지 10%는 그대로 둡니다 (마스킹하지 않습니다)

        padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        labels = torch.tensor(node_labels.tolist() + edge_labels.tolist())
        padded_labels[:len(labels)] = labels



        # padded_labels = torch.zeros(self.max_length, dtype=torch.long)

        # labels = torch.cat((edge_labels, node_labels))

        # padded_labels[:len(labels)] = labels

        return nids, eids, padded_labels

    