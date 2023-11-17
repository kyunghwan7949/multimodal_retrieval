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
        self.mlm = mlm
        self.transforms = transforms
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length

        # # COCO 
        self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        self.img_path = op.join(self.dataset_dir, 'images')
        self.graph_path = f'/data/data2/IRRA/COCO/cs_graph_train_v3'
        self.cache_path = f'/data/data2/IRRA/COCO/cache'
        self.annotation  = json.load(open(self.anno_path, "r"))
        self._add_instance_ids()
        

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1  

        self._add_instance_ids()

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
    
    def random_image_mask(self):
        mask = np.hstack([
            np.zeros(196 - 147), # UNMASK | num_patches, num+mask
            np.ones(147),        # MASK   | num_mask
        ])
        tmp = mask
        np.random.shuffle(mask) # 랜덤
        return mask # (196,)
    
    def __getitem__(self, index):
        
        ## COCO
        caption, image_path, image_id, pid = self.annotation[index].values()
        image = self._load_image(image_path)
        pid = torch.tensor(int(pid))

        image_tmp = self._load_image(image_path)
        
        if self.transforms is not None:
            image = self.transforms(image) # 

        graph = self._load_graph(pid)
        
        image_mask_np = self.random_image_mask()
        image_mask = torch.tensor(image_mask_np).view(1, -1)
     
        node_input_ids = self.get_node_info(graph['node_tokens'])

        # edge_tokens = []
        # for token in graph['edge_tokens']:
        #     edge_tokens.append(token)
        #     edge_tokens.append(token)
        # edge_input_ids = torch.cat([torch.tensor(self.tokenizer.encode(edge)) for edge in edge_tokens])
        edge_index = graph['edge_index']
        if node_input_ids.size(0) > self.max_length:
            edge_index = edge_index[:, :self.max_length]
            node_input_ids = node_input_ids[:self.max_length]

        ## 노드 마스킹 
        # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
        #     edge_input_ids = edge_input_ids[:self.max_length - node_input_ids.size(0)]
        #     edge_index = edge_index[:, :self.max_length - node_input_ids.size(0)]

        # lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)
        
        ## 엣지 마스킹
        # 엣지가 짝수여야하는데, edge가 max_length로 홀수로 짤릴수 있기 때문에 -1 추가
        # if node_input_ids.size(0) + edge_input_ids.size(0) > self.max_length:
        #     elength = self.max_length - node_input_ids.size(0) 
        #     if elength % 2 != 0 :
        #         elength-=1
        #     edge_input_ids = edge_input_ids[:elength]
        #     edge_index = edge_index[:, :elength]
            
        lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

        caption_ids = torch.zeros(self.max_length,dtype=torch.long)
        caption_ids[:node_input_ids.size(0)] = node_input_ids
        
        mlm_node, mlm_label = self._masked_node(graph, node_input_ids, self.tokenizer)

        lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)
        
        return {
            "pids": pid,
            "image_ids": image_id,
            "images": image,
            "caption_ids": caption_ids,
            "node_data": node_input_ids,
            # "edge_data": edge_input_ids,
            "edge_index": edge_index,
            "lap_eigvec": lap_eigvec.half(),
            "lap_eigval": lap_eigval.half(),
            "mlm_node": mlm_node,
            # "mlm_edge": mlm_edge,
            "mlm_label" : mlm_label,
            "image_mask" : image_mask
        }
        
    ## Sub Graph Masking
    def _masked_node(self, graph, node_input_ids, tokenizer):
        
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 1 ~ 49405
        
        nids = node_input_ids.clone()
        # print("마스킹전 : ",nids)

        fin_list = []
        for item in graph['first_ancestor_info']:
            if item[1] == 'NP':
                fin_item = item[0]
                fin_list.append(fin_item)
        # print(fin_list)
        
        fin_list_indices = []
        for item in fin_list:
            # 'a'와 같은 항목이 여러 번 등장하는 경우, 각 등장에 대해 index를 따로 탐색
            indices = [i for i, x in enumerate(graph['node_tokens']) if x == item]
            # fin_list_indices에 아직 추가되지 않은 index만 추가
            for index in indices:
                if index not in fin_list_indices:
                    fin_list_indices.append(index)
                    break  # 첫번쨰 "NP" 조상노드 찾으면 반복문 종료
                
        # 마스킹할 노드 중 50%(masked_ratio) 노드만 선택
        masked_ratio = 0.8
        
        masked_node_selected = []
        while len(masked_node_selected) < int(len(fin_list_indices) * masked_ratio):
            random_index = random.randint(0, len(fin_list_indices) - 1)
            element = fin_list_indices[random_index]
            if element not in masked_node_selected:
                masked_node_selected.append(element)

        # masked_node_list = torch.tensor(fin_list_indices).clone()
        masked_node_list = torch.tensor(masked_node_selected).clone()
        # print(masked_node_list)
        # print("--------------")
        
        original_values = []
        original_indices = []
        
        for i in masked_node_list:
            for j, data in enumerate(nids):
                if i == j:
                    original_values.append(nids[i].item())
                    original_indices.append(i.item())
                    nids[i] = mask
                    
        # print("마스킹후 : ",nids)
        node_labels = nids.clone()
        node_labels[nids != mask] = 0

        for idx, val in zip(original_indices, original_values):
            node_labels[idx] = val
            
        # print("node_label : ",node_labels)

        # if all(l == 0 for l in node_labels):
        #     # at least mask 1
        #     node_labels[0] = nids[0]
        #     nids[0] = mask
        
        padded_labels = torch.zeros(self.max_length, dtype=torch.long) 
        labels = torch.tensor(node_labels)
        padded_labels[:len(labels)] = labels
        
        # nids : 마스킹 후 노드 ID  -> [노드 갯수]
        # padded_labels : 마스킹 된 노드 ID, 나머지 모두 0 -> [max_length=77]
        
        a = nids
        b = padded_labels
        
        return nids, padded_labels   

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
        self.graph_path = f'/data/data2/IRRA/COCO/cs_graph_val_v3'
        self.cache_path = f'/data/data2/IRRA/COCO/cache'
        self.annotation  = json.load(open(self.anno_path, "r"))

        self.transforms = transforms
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length

        self._add_instance_ids()

        self.graph = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
    
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            
            self.image.append(ann["image"])
            image_orig = self._load_image(ann["image"])
            image_t = self.transforms(image_orig)
            
            image_mask_np = self.random_image_mask()
            image_mask = torch.tensor(image_mask_np).view(1, -1)
            
            self.img2txt[img_id] = []

            for i, caption in enumerate(ann["caption"]):
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id

                graph_dict = {}
                
                
                # image_path = os.path.join(self.img_path, self.annotation[img_id]["image"])
        
                # image = Image.open(image_path).convert("RGB")
                # image = self.transforms(image)
                
                # image_mask_np = self.random_image_mask()
                # image_mask = torch.tensor(image_mask_np).view(1, -1)

                graph = self._load_graph(txt_id)

                node_input_ids = self.get_node_info(graph['node_tokens'])

                edge_index = graph['edge_index']
                
                if node_input_ids.size(0) > self.max_length:
                    edge_index = edge_index[:, :self.max_length]
                    node_input_ids = node_input_ids[:self.max_length]

                lap_eigvec, lap_eigval = preprocess_item(node_input_ids.size(0), edge_index)

                caption_ids = torch.zeros(self.max_length, dtype=torch.long)
                caption_ids[:node_input_ids.size(0)] = node_input_ids             
                  
                mlm_node, mlm_label = self._masked_node(graph, node_input_ids, self.tokenizer)
                
                graph_dict["node_data"] = node_input_ids
                # graph_dict["edge_data"] = edge_input_ids
                graph_dict["caption_ids"] = caption_ids
                graph_dict["edge_index"] = edge_index
                
                graph_dict["mlm_node"] = mlm_node
                # graph_dict["mlm_edge"] = mlm_edge
                graph_dict["mlm_label"] = mlm_label
                
                graph_dict["lap_eigvec"] = lap_eigvec.half()
                graph_dict["lap_eigval"] = lap_eigval.half()
                
                graph_dict["images"] = image_t.half()
                graph_dict["image_mask"] = image_mask.half()
                            
                self.graph.append(graph_dict)
                txt_id += 1
            T=3
        t=3
    
    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def _load_image(self, path):
        return Image.open(os.path.join(self.img_path, path)).convert("RGB")

    def _load_graph(self, pid):
        path = f'{self.split}_graph_{pid}.pt'
        return torch.load(op.join(self.graph_path, path))
    
    def random_image_mask(self):
        mask = np.hstack([
            np.zeros(196 - 147), # UNMASK | num_patches, num+mask
            np.ones(147),        # MASK   | num_mask
        ])
        tmp = mask
        np.random.shuffle(mask) # 랜덤
        return mask # (196,)
    
    # def __getitem__(self, index):
    #     ab = self.annotation[index]
    #     # COCO
    #     image_path = os.path.join(self.img_path, self.annotation[index]["image"])
        
    #     image = Image.open(image_path).convert("RGB")
    #     image_t = self.transforms(image)
        
    #     image_mask_np = self.random_image_mask()
    #     image_mask = torch.tensor(image_mask_np).view(1, -1)
        
    #     return {
    #             "image": image_t, 
    #             "index": index,
    #             "iamge_mask" : image_mask
    #             }
    def __getitem__(self, index):
        ab = self.annotation[index]
        # COCO
        image_path = os.path.join(self.img_path, self.annotation[index]["image"])
        
        image = Image.open(image_path).convert("RGB")

        image = self.transforms(image)
        
        image_t = self.transforms(image)

        return {"image": image, "index": index, "image_t":image_t}
    
    
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

     ## Sub Graph Masking
    def _masked_node(self, graph, node_input_ids, tokenizer):
        
        mask = tokenizer.encoder["<mask>"]
        token_range = list(range(4, len(tokenizer.encoder)-1)) # 1 ~ 49405
        
        nids = node_input_ids.clone()
        # print("마스킹전 : ",nids)

        fin_list = []
        for item in graph['first_ancestor_info']:
            if item[1] == 'NP':
                fin_item = item[0]
                fin_list.append(fin_item)
        # print(fin_list)
        
        fin_list_indices = []
        for item in fin_list:
            # 'a'와 같은 항목이 여러 번 등장하는 경우, 각 등장에 대해 index를 따로 탐색
            indices = [i for i, x in enumerate(graph['node_tokens']) if x == item]
            # fin_list_indices에 아직 추가되지 않은 index만 추가
            for index in indices:
                if index not in fin_list_indices:
                    fin_list_indices.append(index)
                    break  # 첫번쨰 "NP" 조상노드 찾으면 반복문 종료
                
        # 마스킹할 노드 중 50%(masked_ratio) 노드만 선택
        masked_ratio = 0.8
        
        masked_node_selected = []
        while len(masked_node_selected) < int(len(fin_list_indices) * masked_ratio):
            random_index = random.randint(0, len(fin_list_indices) - 1)
            element = fin_list_indices[random_index]
            if element not in masked_node_selected:
                masked_node_selected.append(element)

        # masked_node_list = torch.tensor(fin_list_indices).clone()
        masked_node_list = torch.tensor(masked_node_selected).clone()
        # print(masked_node_list)
        # print("--------------")
        
        original_values = []
        original_indices = []
        
        for i in masked_node_list:
            for j, data in enumerate(nids):
                if i == j:
                    original_values.append(nids[i].item())
                    original_indices.append(i.item())
                    nids[i] = mask
                    
        # print("마스킹후 : ",nids)
        node_labels = nids.clone()
        node_labels[nids != mask] = 0

        for idx, val in zip(original_indices, original_values):
            node_labels[idx] = val
            
        # print("node_label : ",node_labels)

        # if all(l == 0 for l in node_labels):
        #     # at least mask 1
        #     node_labels[0] = nids[0]
        #     nids[0] = mask
        
        padded_labels = torch.zeros(self.max_length, dtype=torch.long) 
        labels = torch.tensor(node_labels)
        padded_labels[:len(labels)] = labels
        
        # nids : 마스킹 후 노드 ID  -> [노드 갯수]
        # padded_labels : 마스킹 된 노드 ID, 나머지 모두 0 -> [max_length=77]
        
        a = nids
        b = padded_labels
        
        return nids, padded_labels