import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
# from torch import optim
from torch.utils.data import DataLoader
import torchtext
import dgl
import os
import tqdm
import pandas as pd

import layers
import sampler as sampler_module
import evaluation
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

def train(dataset, args):
    g = dataset['train-graph']
    # test_matrix = dataset['test-matrix'].tocsr()
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']

    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(g.num_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(g.num_nodes(item_ntype))

    # Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = sampler_module.PinSAGECollator(neighbor_sampler, g, item_ntype)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(g, item_ntype, args.hidden_dims, args.num_layers).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = optim.lr_scheduler.StepLR(opt, step_size=2500, gamma=0.1)

    # For each batch of head-tail-negative triplets...
    cost_evol = []
    for epoch_id in range(args.num_epochs):
        print(f'Epoch {epoch_id}/{args.num_epochs}')
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()

            if batch_id == args.batches_per_epoch-1:
              cost_evol.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
          
        # #Decay lr  
        # scheduler.step()
        # print(f'LR:{opt.param_groups[0]["lr"]}')

        # Evaluate
        if ((epoch_id % 1000 == 0) and (epoch_id != 0)):
          model.eval()
          with torch.no_grad():
              item_batches = torch.arange(g.num_nodes(item_ntype)).split(args.batch_size)
              h_item_batches = []
              for blocks in dataloader_test:
                  for i in range(len(blocks)):
                      blocks[i] = blocks[i].to(device)

                  h_item_batches.append(model.get_repr(blocks))
              h_item = torch.cat(h_item_batches, 0)

              # print(evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size))
              torch.save(h_item, f"embeddings_epochs//lr{args.lr}_hd{args.hidden_dims}_nn{args.num_neighbors}_epoch{args.epoch_id}.pth")
              
    cost_evol = pd.DataFrame(cost_evol)
    cost_evol.to_csv('Cost_Evolution.csv')

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.50)
    parser.add_argument('--num-random-walks', type=int, default=50)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=256) #256
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')        # can also be "cuda:0"
    parser.add_argument('--num-epochs', type=int, default=5000)
    parser.add_argument('--batches-per-epoch', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    data_info_path = os.path.join(args.dataset_path, 'data.pkl')
    with open(data_info_path, 'rb') as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, 'train_g.bin')
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset['train-graph'] = g_list[0]
    train(dataset, args)
