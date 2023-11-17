import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter
import json
import torch
import os.path as op

## COCO
class Vocabulary(object):

    def __init__(self,
        # vocab_file='/data/data2/IRRA/COCO/vocab.pkl', # COCO DP
        # vocab_file='/data/data2/IRRA/COCO/vocab_cs_fin.pkl', # COCO CS
        vocab_file='/data/data2/IRRA/COCO/vocab_cs_v4.pkl', # COCO CS
        start_word="<s>",
        end_word="</s>",
        unk_word="<unk>",
        mask_word="<mask>",
        pad_word="<pad>",
        train_annotations_file='/data/data2/IRRA/COCO/annotations/coco_karpathy_train.json',
        val_annotations_file='/data/data2/IRRA/COCO/annotations/coco_karpathy_val.json',
        vocab_from_file=True):
 
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.mask_word = mask_word
        self.pad_word = pad_word
        self.train_annotations_file = train_annotations_file
        self.val_annotations_file = val_annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            # print('Vocabulary successfully loaded from vocab.pkl file!') # DP
            print('Vocabulary successfully loaded from vocab_cs_v4.pkl file!') # CS
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
        
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.pad_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()
        self.add_word(self.mask_word)

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def _load_graph(self, pid):
        path = f'{self.split}_graph_{pid}.pt'
        return torch.load(op.join(self.graph_path, path))

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        counter = Counter()
        
        # # DP Graph
        # train = json.load(open(self.train_annotations_file, "r"))
        # train_graph_path = '/data/data2/IRRA/COCO/dp_graph_train'
        # for i, id in enumerate(train):
        #     path = f'train_graph_{i}.pt'
        #     graph =torch.load(op.join(train_graph_path, path))
        #     node_tokens = graph['node_tokens']
        #     edge_tokens = graph['edge_tokens']
        #     train_node = [ntoken.lower() for ntoken in node_tokens]
        #     train_edge = [etoken.lower() for etoken in edge_tokens]
        #     counter.update(train_node)
        #     counter.update(train_edge)

        # val = json.load(open(self.val_annotations_file, "r"))
        # val_graph_path = '/data/data2/IRRA/COCO/dp_graph_val'
        
        # txt_id = 0
        # for img_id, ann in enumerate(val):
        #     for _, _ in enumerate(ann["caption"]):
        #         path = f'val_graph_{txt_id}.pt'
        #         graph =torch.load(op.join(val_graph_path, path))
            
        #         edge_tokens = graph['edge_tokens']
        #         node_tokens = graph['node_tokens']

        #         val_edge = [etoken.lower() for etoken in edge_tokens]
        #         val_node = [ntoken.lower() for ntoken in node_tokens]
                
        #         counter.update(val_edge)
        #         counter.update(val_node)
        #         txt_id += 1
                
         #####################################################################          
        # CS Graph
        train = json.load(open(self.train_annotations_file, "r"))
        train_graph_path ='/data/data2/IRRA/COCO/cs_graph_train_v3'
        for i, id in enumerate(train):
            path = f'train_graph_{i}.pt'
            graph =torch.load(op.join(train_graph_path, path))
            
            node_tokens = graph['node_tokens']
            # edge_tokens = graph['edge_tokens']
            train_node = [ntoken.lower() for ntoken in node_tokens]
            # train_edge = [etoken.lower() for etoken in edge_tokens]
            counter.update(train_node)
            # counter.update(train_edge)

        val = json.load(open(self.val_annotations_file, "r"))
        val_graph_path = '/data/data2/IRRA/COCO/cs_graph_val_v3'
        
        txt_id = 0
        for img_id, ann in enumerate(val):
            for _, _ in enumerate(ann["caption"]):
                path = f'val_graph_{txt_id}.pt'
                graph =torch.load(op.join(val_graph_path, path))
            
                # edge_tokens = graph['edge_tokens']
                node_tokens = graph['node_tokens']
                cons_parse_edge = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X', '.', ',']

                # val_edge = [etoken.lower() for etoken in edge_tokens]
                val_node = [ntoken.lower() for ntoken in node_tokens]
                cons_edge = [etoken.lower() for etoken in cons_parse_edge]
                
                # counter.update(val_edge)
                counter.update(val_node)
                counter.update(cons_edge)
                txt_id += 1
                
                
                
         #####################################################################       
        
        # ## CS Graph
        # train = json.load(open(self.train_annotations_file, "r"))
        # train_graph_path = '/data/data2/IRRA/COCO/cs_graph_train_v3'
        # for i, id in enumerate(train):
        #     path = f'train_graph_{i}.pt'
        #     graph =torch.load(op.join(train_graph_path, path))
        #     node_tokens = graph['node_tokens']
        #     cons_parse_edge = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X', '.', ',']            
            
        #     train_node = [ntoken.lower() for ntoken in node_tokens]
        #     cons_edge = [etoken.lower() for etoken in cons_parse_edge]
            
        #     counter.update(train_node)
        #     counter.update(cons_edge)

        # val = json.load(open(self.val_annotations_file, "r"))
        # val_graph_path = '/data/data2/IRRA/COCO/cs_graph_val_v3'
        
        # for i, id in enumerate(val):
        #     path = f'val_graph_{i}.pt'
        #     graph =torch.load(op.join(val_graph_path, path))
        #     node_tokens = graph['node_tokens']
        #     # cons_parse_edge = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X', '.', ',']            
            
        #     val_node = [ntoken.lower() for ntoken in node_tokens]

        #     # cons_edge = [etoken.lower() for etoken in cons_parse_edge]
            
        #     counter.update(val_node)
        #     # counter.update(cons_edge)
            
        # val_graph_path = '/data/data2/IRRA/COCO/fin_cs_graph_val'
        # val_graph_path = '/mount/IRRA/COCO/dp_graph_val'
        # txt_id = 0
        # for img_id, ann in enumerate(val):
        #     for _, _ in enumerate(ann["caption"]):
        #         path = f'val_graph_{txt_id}.pt'
        #         graph =torch.load(op.join(val_graph_path, path))
            
        #         # edge_tokens = graph['edge_tokens']
        #         node_tokens = graph['node_tokens']

        #         # val_edge = [etoken.lower() for etoken in edge_tokens]
        #         val_node = [ntoken.lower() for ntoken in node_tokens]
                
        #         # counter.update(val_edge)
        #         counter.update(val_node)
        #         txt_id += 1

        words = [word for word, cnt in counter.items()]
        # print(words)

        for i, word in enumerate(words):
            self.add_word(word)        
             
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    

# ## RSTPReid
# class Vocabulary(object):

#     def __init__(self,
#         vocab_file='/data/data2/khahn/MGM_train/data/RSTPReid/rstpreid_fin_vocab.pkl', ## RSTPReid DP
#         # vocab_file='/data/data2/khahn/MGM_train/data/RSTPReid/rstpreid_fin_vocab.pkl', ## RSTPReid CS
#         start_word="<s>",
#         end_word="</s>",
#         unk_word="<unk>",
#         mask_word="<mask>",
#         pad_word="<pad>",
#         # train_annotations_file='/data/data2/khahn/MGM_train/data/RSTPReid/train.json',
#         # val_annotations_file='/data/data2/khahn/MGM_train/data/RSTPReid/val.json',
#         # train_annotations_file='/data/data2/khahn/MGM_train/data/RSTPReid/splited_updated_train.json',
#         # val_annotations_file='/data/data2/khahn/MGM_train/data/RSTPReid/splited_updated_val.json',
#         train_annotations_file='/data/data2/khahn/MGM_train/data/RSTPReid/annotations/merged_train.json',
#         val_annotations_file='/data/data2/khahn/MGM_train/data/RSTPReid/annotations/merged_val.json',
#         vocab_from_file=True
#         ):
        
#         self.vocab_file = vocab_file
#         self.start_word = start_word
#         self.end_word = end_word
#         self.unk_word = unk_word
#         self.mask_word = mask_word
#         self.pad_word = pad_word
#         self.train_annotations_file = train_annotations_file
#         self.val_annotations_file = val_annotations_file
#         self.vocab_from_file = vocab_from_file
#         # self.add_captions()
#         # t=3
#         self.get_vocab()

#     def get_vocab(self):
#         """Load the vocabulary from file OR build the vocabulary from scratch."""
#         if os.path.exists(self.vocab_file) & self.vocab_from_file:
#             # print(self.vocab_file)
#             with open(self.vocab_file, 'rb') as f:
#                 vocab = pickle.load(f)
#                 self.word2idx = vocab.word2idx
#                 self.idx2word = vocab.idx2word
                
#                 # print(self.idx2word.values())
#                 t=3
#                 # print(self.idx2word.keys())
#             print('Vocabulary successfully loaded from rstpreid_fin_vocab.pkl file!')
#         else:
#             self.build_vocab()
#             with open(self.vocab_file, 'wb') as f:
#                 pickle.dump(self, f)
        
#     def build_vocab(self):
#         """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
#         self.init_vocab()
#         self.add_word(self.start_word)
#         self.add_word(self.pad_word)
#         self.add_word(self.end_word)
#         self.add_word(self.unk_word)
#         self.add_captions()
#         self.add_word(self.mask_word)

#     def init_vocab(self):
#         """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
#         self.word2idx = {}
#         self.idx2word = {}
#         self.idx = 0

#     def _load_graph(self, pid):
#         path = f'{self.split}_graph_{pid}.pt'
#         return torch.load(op.join(self.graph_path, path))

#     def add_word(self, word):
#         """Add a token to the vocabulary."""
#         if not word in self.word2idx:
#             self.word2idx[word] = self.idx
#             self.idx2word[self.idx] = word
#             self.idx += 1

#     def add_captions(self):
#         """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
#         counter = Counter()
        
#         # train = json.load(open(self.train_annotations_file, "r"))
#         # train_graph_path = '/data/data2/khahn/MGM_train/data/dp_graph/dp_graph_train_splited_update'

#         # for i, id in enumerate(train):
#         #     path = f'train_graph_{i}.pt'
#         #     graph =torch.load(op.join(train_graph_path, path))
#         #     node_tokens = graph['node_tokens']
#         #     edge_tokens = graph['edge_tokens']
#         #     train_node = [ntoken.lower() for ntoken in node_tokens]
#         #     train_edge = [etoken.lower() for etoken in edge_tokens]
#         #     counter.update(train_node)
#         #     counter.update(train_edge)



#         # val = json.load(open(self.val_annotations_file, "r"))
#         # val_graph_path = '/data/data2/khahn/MGM_train/data/dp_graph/dp_graph_val_splited_update'
        
#         # txt_id = 0
#         # for img_id, ann in enumerate(val):
#         #     for _, _ in enumerate(ann["captions"]):

#         #         path = f'val_graph_{txt_id}.pt'
#         #         graph =torch.load(op.join(val_graph_path, path))
            
#         #         edge_tokens = graph['edge_tokens']
#         #         node_tokens = graph['node_tokens']

#         #         val_edge = [etoken.lower() for etoken in edge_tokens]
#         #         val_node = [ntoken.lower() for ntoken in node_tokens]
                
#         #         counter.update(val_edge)
#         #         counter.update(val_node)
#         #         txt_id += 1
        
#         # train = json.load(open(self.train_annotations_file, "r"))
#         train_graph_path = '/data/data2/khahn/MGM_train/data/RSTPReid/dp_graph/fin_train'

#         # for i, id in enumerate(train):
#         for i in range(80189):
#             path = f'train_graph_{i}.pt'
#             graph =torch.load(op.join(train_graph_path, path))
#             node_tokens = graph['node_tokens']
#             edge_tokens = graph['edge_tokens']
#             train_node = [ntoken.lower() for ntoken in node_tokens]
#             train_edge = [etoken.lower() for etoken in edge_tokens]
#             counter.update(train_node)
#             counter.update(train_edge)
            

#         tmp = []
#         # val = json.load(open(self.val_annotations_file, "r"))
#         val_graph_path = '/data/data2/khahn/MGM_train/data/RSTPReid/dp_graph/fin_val'
        
#         # for i, id in enumerate(val):
#         for i in range(3224):
#             path = f'val_graph_{i}.pt'
#             graph =torch.load(op.join(val_graph_path, path))
#             node_tokens = graph['node_tokens']
#             # print("노드",i, node_tokens)
#             edge_tokens = graph['edge_tokens']
#             val_node = [ntoken.lower() for ntoken in node_tokens]
#             val_edge = [etoken.lower() for etoken in edge_tokens]
#             counter.update(val_node)
#             counter.update(val_edge)
            
#         #     for token in val_node:
               
#         #         if token not in tmp:
#         #             tmp.append(token)
           
#         # print(tmp)
#         # print(len(tmp))
            
#             # if "glassed" in val_node:
#             #     print(f"노드 {i}: {val_node}")
#             # if "glassed" in val_edge:
#             #     print(f"엣지 {i}: {val_edge}")
#             # val_nodesss.append(train_node)
            
#         # print("dddddddddddddddd",counter.keys())
            
#             # if i == 1128:
#             #     print(f"노드 {i}: {val_node}")
#             #     print(f"엣지 {i}: {val_edge}")
#             # # print(i)
        

#         words = [word for word, cnt in counter.items()]

#         for i, word in enumerate(words):
#             self.add_word(word)        
             
#     def __call__(self, word):
#         if not word in self.word2idx:
#             return self.word2idx[self.unk_word]
#         return self.word2idx[word]

#     def __len__(self):
#         return len(self.word2idx)
    
# print("hi")
# if __name__ == '__main__':
#     print("hi")


# tmp = Vocabulary()
# t=3