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
from trainer import Trainer
import os
dataset = "Video_Games"
ckpt_path = f"./results/{dataset}/3x256-0.0-0.0-0.0/Aug-30-2024_16-25-00/best_collision_model.pth"
# ckpt_path = "./results/Video_Games/Aug-16-2024_08-58-50/best_collision_model.pth"
output_dir = f"./ID_generation/preprocessing/processed/{dataset}/"
# output_file = "Games.bertindex.json"
output_file = f"{dataset}.sentence_t5_3x256_test_1.index.json"
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
print(args)
state_dict = ckpt["state_dict"]


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

indices_count = {}
all_indices = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]
# prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>"]


for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
        code_str = str(code)
        # print(code_str)
        if code_str in indices_count:
            code.append(prefix[-1].format(int(indices_count[code_str])))
            indices_count[code_str] += 1
        else:
            code.append(prefix[-1].format(int(0)))
            indices_count[code_str] = 1

        all_indices.append(code)
    # break

# print(all_indices)
print("All indices number: ",len(all_indices))
print("Ratio of unique indices: ", len(indices_count)/len(all_indices))
print("Max number of conflicts: ", max(indices_count.values()))

all_indices_dict = {}
for item, indices in enumerate(all_indices):
    all_indices_dict[item+1] = indices


ss =set()
for key in all_indices_dict:
    for t in all_indices_dict[key]:
        ss.add(t)
print(len(list(ss)))
# print(list(ss)) 


# with open(output_file, 'w') as fp:
#     json.dump(all_indices_dict,fp)

# print(all_indices_dict[1])
# print(all_indices_dict[2])
# print(all_indices_dict[0])
