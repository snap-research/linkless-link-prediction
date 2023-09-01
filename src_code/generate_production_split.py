import random

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges, to_networkx, subgraph)
from torch.nn.functional import one_hot
import math
from ogb.linkproppred import PygLinkPropPredDataset

def add_node_feats(data, device, type='degree'):
    assert (type == 'degree')

    G = to_networkx(data)
    degrees = torch.tensor([v for (_, v) in G.degree()])
    data.x = one_hot(degrees).to(device).float()
    return data

def perform_subset_negative_sampling(node_mask, pos_index):
    device = node_mask.device
    n_selected = node_mask.sum().item()
    # first, create relabeled edge index starting at 0
    new_mapping = torch.zeros(node_mask.size(), dtype=pos_index.dtype, device=device)
    new_mapping[node_mask] = torch.arange(n_selected, device=device)

    inverse_mapping = torch.arange(node_mask.size(0), device=device)[node_mask]
    
    relabeled_index = new_mapping[pos_index]
    neg_relabeled_index = negative_sampling(relabeled_index, n_selected, pos_index.size(1), force_undirected=True)
    return inverse_mapping[neg_relabeled_index]

def perform_subset_bipartite_negative_sampling(node_mask_src, node_mask_dst, pos_index):
    device = node_mask_src.device
    n_selected_src = node_mask_src.sum().item()
    n_selected_dst = node_mask_dst.sum().item()

    # relabel edge index so that we only have edges from old -> new, not new -> old
    reorder_mask = node_mask_dst[pos_index[0, :]]
    old_srcs = pos_index[0, reorder_mask].clone()
    pos_index[0, reorder_mask] = pos_index[1, reorder_mask]
    pos_index[1, reorder_mask] = old_srcs

    # first, create relabeled edge index starting at 0
    new_mapping_src = torch.zeros(node_mask_src.size(), dtype=pos_index.dtype, device=device)
    new_mapping_src[node_mask_src] = torch.arange(n_selected_src, device=device)
    inverse_mapping_src = torch.arange(node_mask_src.size(0), device=device)[node_mask_src]

    new_mapping_dst = torch.zeros(node_mask_dst.size(), dtype=pos_index.dtype, device=device)
    new_mapping_dst[node_mask_dst] = torch.arange(n_selected_dst, device=device)
    inverse_mapping_dst = torch.arange(node_mask_dst.size(0), device=device)[node_mask_dst]
    
    relabeled_index_src = new_mapping_src[pos_index[0]]
    relabeled_index_dst = new_mapping_dst[pos_index[1]]
    relabeled_index = torch.vstack((relabeled_index_src, relabeled_index_dst))

    neg_relabeled_index = negative_sampling(relabeled_index, (n_selected_src, n_selected_dst), pos_index.size(1))
    return torch.vstack((inverse_mapping_src[neg_relabeled_index[0]], inverse_mapping_dst[neg_relabeled_index[1]]))

def create_mask(base_mask, rows, cols):
    return base_mask[rows] & base_mask[cols]

def split_edges(edge_index, val_ratio, test_ratio):
    mask = edge_index[0] <= edge_index[1]
    perm = mask.nonzero(as_tuple=False).view(-1)
    perm = perm[torch.randperm(perm.size(0), device=perm.device)]
    num_val = int(val_ratio * perm.numel())
    num_test = int(test_ratio * perm.numel())
    num_train = perm.numel() - num_val - num_test
    train_edges = perm[:num_train]
    val_edges = perm[num_train:num_train + num_val]
    test_edges = perm[num_train + num_val:]
    train_edge_index = edge_index[:,train_edges]
    train_edge_index = torch.cat([train_edge_index, train_edge_index.flip([0])], dim=-1)
    val_edge_index = edge_index[:,val_edges]
    val_edge_index = torch.cat([val_edge_index, val_edge_index.flip([0])], dim=-1)
    test_edge_index = edge_index[:,test_edges]

    return train_edge_index, val_edge_index, test_edge_index

def do_production_edge_split(dataset: Dataset, data_name, test_ratio, val_node_ratio, val_ratio, old_old_extra_ratio, split_seed=234):
    # Seed our RNG
    random.seed(split_seed)
    torch.manual_seed(split_seed) 

    # Assume we only have 1 graph in our dataset
    assert(len(dataset) == 1)
    data = dataset[0]

    # Some assertions to help with type inference
    assert(isinstance(data, Data))
    assert(data.num_nodes is not None)

    # sample some negatives to use globally
    num_negatives = round(test_ratio * data.edge_index.size(1)/2)
    negative_samples = negative_sampling(data.edge_index, data.num_nodes, num_negatives, force_undirected=True)

    # Step 1: pick a set of nodes to remove
    node_splitter = RandomNodeSplit(num_val=0.0, num_test=val_node_ratio)
    new_data = node_splitter(data)
    
    # Step 2: Split the edges connecting old-old nodes for training, inference and testing
    rows, cols = new_data.edge_index
    old_old_edges = create_mask(new_data.train_mask, rows, cols)
    old_old_ei = new_data.edge_index[:, old_old_edges]
    old_old_train, old_old_val, old_old_test = split_edges(old_old_ei, old_old_extra_ratio, test_ratio)

    # Step 3: Split the edges connecting old-new nodes for inference and testing
    old_new_edges = (new_data.train_mask[rows] & new_data.test_mask[cols]) | (
        new_data.test_mask[rows] & new_data.train_mask[cols])
    old_new_ei = new_data.edge_index[:, old_new_edges]
    old_new_train, _, old_new_test = split_edges(old_new_ei, 0.0, test_ratio)

    # Step 4: Split the edges connecting new-new nodes for inference and testing
    new_new_edges = create_mask(new_data.test_mask, rows, cols)
    new_new_ei = new_data.edge_index[:, new_new_edges]
    new_new_train, _, new_new_test = split_edges(new_new_ei, 0.0, test_ratio)

    # Step 5: Merge testing edges
    test_edge_index = torch.cat([old_old_test, old_new_test, new_new_test], dim=-1)
    test_edge_bundle = (old_old_test, old_new_test, new_new_test, test_edge_index)

    # Step 6: Prepare the graph for training
    training_only_ei = subgraph(new_data.train_mask, old_old_train, relabel_nodes=True)[0]
    training_only_x = new_data.x[new_data.train_mask]
 
    # Step 7: Generate training/validation set
    given_data = Data(training_only_x, training_only_ei)
    val_splitter = RandomLinkSplit(0.0, val_ratio, is_undirected=True)
    training_data, _, val_data = val_splitter(given_data)

    # Step 8: Merge the edges for inference
    inference_edge_index = torch.cat([old_old_train, old_old_val, old_new_train, new_new_train], dim=-1)
    inference_data = Data(new_data.x, inference_edge_index)

    print("Datasets Infomation:\t\n")
    print("Name:\t" + data_name + "\n")
    print("#Old Nodes:\t" + str(training_only_x.size(0))+"\n")
    print("#New Nodes:\t" + str(new_data.x.size(0) - training_only_x.size(0))+"\n")
    print("#Old-Old testing edges:\t" + str(old_old_test.size(1))+"\n")
    print("#Old-New testing edges:\t" + str(old_new_test.size(1))+"\n")
    print("#New-New testing edges:\t" + str(new_new_test.size(1))+"\n")

    return training_data, val_data, inference_data, data, test_edge_bundle, negative_samples

# From the OGB implementation of SEAL
def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1, split_seed=234):
    data = dataset[0]
    random.seed(split_seed)
    torch.manual_seed(split_seed)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(edge_index,
                                                      num_nodes=data.num_nodes,
                                                      num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(data.edge_index, num_nodes=num_nodes, num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge

if __name__ == "__main__":
    from utils import get_dataset

    ## the dataset for splitting
    dataset = "citeseer"
    ## testing edges ratio (0.3 for cora and citeseer, 0.1 for other datasets)
    test_ratio=0.3
    ## New nodes ratio (0.3 for cora and citeseer, 0.1 for other datasets)
    val_node_ratio=0.3
    ## validation/training splitting ratio (0.3 for cora and citeseer, 0.1 for other datasets)
    val_ratio=0.3 
    ## Splitting ratio for new old-old edges appearing for the inference(0.1 for all datasets)
    old_old_extra_ratio= 0.1 
    
    dset = get_dataset('../data', dataset)
    all_data = do_production_edge_split(dset, dataset, test_ratio, val_node_ratio, val_ratio, old_old_extra_ratio)

    torch.save(all_data, "../data/" + dataset + "_production.pkl")
    