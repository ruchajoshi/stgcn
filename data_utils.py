#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader, Batch


class PeMSD7M(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PeMSD7M, self).__init__(root, transform, pre_transform, pre_filter)
        
        # load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['V_228.csv', 'W_228.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        pemsd7_v = pd.read_csv(self.raw_paths[0], header=None)
        pemsd7_w = pd.read_csv(self.raw_paths[1], header=None)
        pemsd7_v = pemsd7_v.to_numpy()
        pemsd7_w = pemsd7_w.to_numpy()
        data_list = []
        
        nodes = torch.tensor([i for i in range(228)],
                     dtype=torch.float).unsqueeze(1)
        
        edge_index = torch.tensor([[j, i] for j in range(228) for i in range(228) if i!=j],
                   dtype=torch.long)
        
        edge_weight = []
        for k in edge_index:
            i, j = k
            edge_weight = np.append(edge_weight, pemsd7_w[i, j])
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1)
        length = pemsd7_v.shape[0]
        for i in range(length):
            node_feature = torch.tensor(pemsd7_v[i], dtype=torch.float).unsqueeze(1)
            data = Data(x=node_feature,
                        # node_feature=node_feature,
                        edge_index = edge_index.T,
                        edge_weight = edge_weight)
            data_list.append(data)
        # print('In process 2')
        # print(len(data_list))
        self.data, self.slices = self.collate(data_list[0:2880])
        # print('In process 3')
        torch.save((self.data, self.slices), self.processed_paths[0])
        # print('In process 4')


class CustomBatch(Data):
    def __init__(self):
        super(CustomBatch, self).__init__(self)
        self.x = None
        self.edge_index = None
        self.edge_weight = None
        self.batch = None
        self.label = None

    @classmethod
    def from_data_list(self, datalist):
        self.x = torch.cat([data.x for data in datalist[:-1]])
        
        self.batch = torch.tensor([i for i in range(len(datalist[:-1])) for nodes in range(datalist[i].x.shape[0])])

        edge_index = []
        for i, data in enumerate(datalist[:-1]):
            edge_index.append(data.edge_index + i * datalist[i-1].x.shape[0])
        edge_index = torch.reshape(torch.cat(edge_index, 1), (2, -1))
        self.edge_index = edge_index

        self.edge_weight = torch.reshape(torch.cat([data.edge_weight for data in datalist[:-1]]), (-1, 1))

        self.label = datalist[-1].x
        return Data(x=self.x, edge_index=self.edge_index, edge_weight=self.edge_weight, batch=self.batch, label=self.label)


class CustomDataLoader():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.len_of_dataset = len(dataset)
        self.current_batch_number = 0

    def __iter__(self):
        self.current_batch_number = 0
        return self

    def __next__(self):
        if self.current_batch_number < self.len_of_dataset - self.batch_size:
            batch = CustomBatch()
            batch = batch.from_data_list(datalist=self.dataset[self.current_batch_number:(self.current_batch_number + self.batch_size + 1)])
            self.current_batch_number += 1
            return batch
        else:
            raise StopIteration


def load_data(dataset, batch_size=12, train=80, val=10, test=10):
    total = (train+val+test)*1.0
    # print(train/total)
    train, val, test = int(train*100/total), int(val*100/total), int(test*100/total)
    # print(train, val, test)
    percentage_length = int(len(dataset)*0.01)
    # print(percentage_length)

    train_dataset = dataset[:percentage_length*train]
    val_dataset = dataset[percentage_length*train:percentage_length*(train+val)]
    test_dataset = dataset[percentage_length*(train+val):]
    percentage_length, len(train_dataset), len(val_dataset), len(test_dataset)
    # batch_size = 64
    train_loader = CustomDataLoader(train_dataset, batch_size=batch_size)
    val_loader = CustomDataLoader(val_dataset, batch_size=batch_size)
    test_loader = CustomDataLoader(test_dataset, batch_size=batch_size)
    # print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader


def describe_data(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('==================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===============================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Number of training nodes: {data.train_mask.sum()}')
    # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.3f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')