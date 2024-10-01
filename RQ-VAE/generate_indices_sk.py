import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from embdatasets import EmbDataset
from models.rqvae import RQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

# ckpt_path = "/mnt/zhengbowen/rqvae_ckpt/BERT/mean-nokm/best_collision_model.pth"
# ckpt_path = "/mnt/zhengbowen/rqvae_ckpt/LLaMA/Sep-01-2023_10-28-14/epoch_4349_collision_0.0268_model.pth"
# ckpt_path = "/mnt/zhengbowen/rqvae_ckpt/LLaMA/32d-nosk/best_collision_model.pth"
# output_dir = "/mnt/zhengbowen/data/Amazon2018/Processed/Games/"
dataset = "Video_Games"
ckpt_path = "./results_Video_Games/Jul-14-2024_14-30-17/best_collision_model.pth"
output_dir = f"./ID_generation/preprocessing/processed/{dataset}/"
# output_file = "Games.bertindex.json"
# output_dir = f"/media/zhengbowen/data/Amazon2018/Processed/{dataset}/"
# output_file = "Games.bertindex.json"
output_file = f"{dataset}.index.json"
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]
# print(args.data_path)

data = EmbDataset(args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  # sk_epsilons=[0,0,0,0.003],
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)
# print(data[0])
# print(data_loader[0][0])

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
while True:
    if tt >= 11 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    # print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)
        # print(d)
        # d = torch.stack(d, dim=0).to(device)

        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1



# print(all_indices)
print("All indices number: ",len(all_indices))
# print("Ratio of unique indices: ", len(all_indices_str)/len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)


# print(all_indices[[[45, 3302, 6756, 11880, 14662]]])

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)

# print(all_indices_dict[1])
# print(all_indices_dict[2])
# print(all_indices_dict[0])
