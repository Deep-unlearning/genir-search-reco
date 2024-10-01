import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from prompt import sft_prompt, all_prompt
from transformers import T5Tokenizer
from datasets import load_dataset, load_from_disk



class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix
        self.model_name = args.base_model
        self.add_prefix_his = args.add_prefix_his

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train", mode_dataset=None,
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.mode_dataset = mode_dataset


        # load data
        self._load_data()
        
        
        # load data
        if self.mode == 'processed':
            self.inter_data = self._process_data()
        else:
            self._remap_items()
            if self.mode == 'train':
                self.inter_data = self._process_train_data()
            elif self.mode == 'valid':
                self.inter_data = self._process_valid_data()
            elif self.mode == 'test':
                self.inter_data = self._process_test_data()
            else:
                raise NotImplementedError



    def _load_data(self):
        
        if self.mode_dataset is not None:
            with open(os.path.join(self.data_path, self.dataset + self.mode_dataset + ".inter.json"), 'r') as f:
                self.inters = json.load(f)
        else:
            with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
                self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):

        id2code = dict()
        for key,value in self.indices.items():
            new_value = ''.join(value)
            id2code[key] = new_value

        inter_data = []
        
        for user, seq in self.inters:
            user = str(user)
            seq = [id2code[str(item)] for item in seq]
            one_data = dict()
            one_data["item"] = seq[-1]
            history = seq[:-1]
            if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
            one_data["inters"] = self.his_sep.join(history)
            # Add "Predict next item" prompt
            if self.add_prefix_his:
                one_data["inters"] = "Predict next item: " + one_data["inters"]
            inter_data.append(one_data)
        
        return inter_data

    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data
    
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]

        return dict(input_ids=d["inters"], labels=d["item"])


class ItemFeatDataset(BaseDataset):

    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        # load data
        self._load_data()
        self.feat_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".data_maps"), 'r') as f:
            self.item_feat = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):

        id2code = dict()
        for key,value in self.indices.items():
            new_value = ''.join(value)
            id2code[key] = new_value

        feat_data = []
        for k,v in self.item_feat['id2meta'].items():
            # print(k)
            # Skip '0' key
            if k != '0':               
                feat = dict()
                index = id2code[str(k)]
                feat["item"] = index
                feat["description"] = v.strip().strip(".!?,;:`")
                # print(feat)
                feat_data.append(feat)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data


    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        # response = response[:1024]
        input = input[:2048]
        return input, response

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)


class ProductSearchDataset(BaseDataset):

    def __init__(self, args, mode="train", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num
        self.mode = mode

        self.prompts = all_prompt["productsearch"]

        # load data
        self._load_data()
        self.feat_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".data_maps"), 'r') as f:
            self.item_feat = json.load(f)
        self.product_search_queries = load_from_disk(os.path.join(self.data_path, self.dataset + "_esci"))
        self.product_search_queries = self.product_search_queries[self.mode]
        if self.mode == "train":
            self.product_search_queries_c4 = load_from_disk(os.path.join(self.data_path, self.dataset + "_c4"))
            self.product_search_queries_c4 = self.product_search_queries_c4['test']
        else:
            self.product_search_queries_c4 = []


    def _process_data(self):
        id2code = dict()
        for key,value in self.indices.items():
            new_value = ''.join(value)
            id2code[key] = new_value

        feat_data = []
        for query in self.product_search_queries:
            feat = dict()
            feat['query'] = query['query']
            item_id = self.item_feat['item2id'][query['product_id']]
            feat['item'] = id2code[str(item_id)]
            feat_data.append(feat)  

        for query in self.product_search_queries_c4:
            feat = dict()
            feat['query'] = query['query']
            item_id = self.item_feat['item2id'][query['item_id']]
            feat['item'] = id2code[str(item_id)]
            feat_data.append(feat)  
        return feat_data
    
    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        data = self.feat_data[idx]
        prompt = random.choice(self.prompts)

        query = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        return dict(input_ids=query, labels=response)