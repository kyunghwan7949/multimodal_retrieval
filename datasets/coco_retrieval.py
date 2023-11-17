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

class RetrievalDataset(Dataset):

    def __init__(self, args, split, transforms, mlm=False):
        super(RetrievalDataset, self).__init__()
        
        self.args = args
        self.dataset_dir = op.join(self.args.root_dir, self.args.dataset_name)
        self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        self.img_path = op.join(self.dataset_dir, 'images')

        self.mlm = mlm
        self.transforms = transforms
        self.tokenizer = SimpleTokenizer()

        self.annotation  = json.load(open(self.anno_path, "r"))[:5]
        self._add_instance_ids()

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def _load_image(self, path):
        return Image.open(os.path.join(self.img_path, path)).convert("RGB")

    def __getitem__(self, index):
        
        caption, image_path, image_id, pid = self.annotation[index].values()
        image = self._load_image(image_path)
        pid = int(pid)

        if self.transforms is not None:
            image = self.transforms(image)

        caption_ids = tokenize(caption, tokenizer=self.tokenizer)
        if self.mlm:
            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_ids.cpu().numpy(),self.tokenizer)

            return {
                "pids": pid,
                "images": image,
                "caption_ids": caption_ids,
                "mlm_ids": mlm_tokens,
                "mlm_labels": mlm_labels
            }
        
        else:
            return {
                "pids": pid,
                "image_ids": image_id,
                "images": image,
                "caption_ids": caption_ids,
            }

    def __len__(self) -> int:
        return len(self.annotation)

    def _build_random_masked_tokens_and_labels(self, tokens, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

   
class RetrievalEvalDataset(Dataset):

    def __init__(self, args, split, transforms):
        super(RetrievalEvalDataset, self).__init__()
        
        self.args = args
        self.dataset_dir = op.join(self.args.root_dir, self.args.dataset_name)
        self.anno_path = op.join(self.dataset_dir, 'annotations', f'coco_karpathy_{split}.json')
        self.img_path = op.join(self.dataset_dir, 'images')

        self.transforms = transforms
        self.tokenizer = SimpleTokenizer()

        self.annotation  = json.load(open(self.anno_path, "r"))
        self._add_instance_ids()

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
    
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                caption_ids = tokenize(caption, tokenizer=self.tokenizer)
                self.text.append(caption_ids)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
        
    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def _load_image(self, path):
        return Image.open(os.path.join(self.img_path, path)).convert("RGB")

    def __getitem__(self, index):

        image_path = os.path.join(self.img_path, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.transforms(image)

        return {"image": image, "index": index}
    
    def __len__(self) -> int:
        return len(self.annotation)
    
