import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

from utils import get_dataset, do_edge_split, randomly_drop_nodes
from torch.nn import BCELoss

from models import MLP, GCN, SAGE, LinkPredictor, GAT, APPNP_model
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists

from utils import do_edge_split_weibo

# from torch_geometric.transforms import RandomLinkSplit

def train(model, predictor, data, split_edge, optimizer, batch_size, encoder_name, dataset):
    # if dataset != "collab":
    #     row, col = data.adj_t
    # elif dataset == "collab":
    #     row, col,_ = data.adj_t.coo()
    row, col = data.adj_t
    
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    criterion = BCELoss()
    # criterion = torch.nn.MSELoss()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        if encoder_name == 'mlp':
            h = model(data.x)
        else:
            h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()

        # if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
        if dataset != "collab" and dataset != "ppa":
            neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')
        elif dataset == "collab" or dataset == "ppa":
            neg_edge = torch.randint(0, data.x.size()[0], edge.size(), dtype=torch.long,
                             device=h.device)

        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        loss = criterion(out, train_label)

        # pos_out = predictor(h[edge[0]], h[edge[1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        # neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]])
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        # loss = pos_loss + neg_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
        # num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset):
    model.eval()
    predictor.eval()

    if encoder_name == 'mlp':
        h = model(data.x)
    else:
        h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    # if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
    if dataset != "collab" and dataset != "ppa":
        neg_train_edge = split_edge['train']['edge_neg'].to(h.device)
    elif dataset == "collab" or dataset == "ppa":
        neg_train_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    # if encoder_name == 'mlp':
    #     h = model(data.x)
    # else:
    #     h = model(data.x, data.full_adj_t)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    # if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
    for K in [10, 50, 100, 200]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    train_result = torch.cat((torch.ones(pos_train_pred.size()), torch.zeros(neg_train_pred.size())), dim=0)
    train_pred = torch.cat((pos_train_pred, neg_train_pred), dim=0)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(train_result.cpu().numpy(),train_pred.cpu().numpy()), roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()),roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()))

    return results

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--datasets', type=str, default='weibo')
    parser.add_argument('--predictor', type=str, default='mlp')  ##inner/mlp
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@200', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets == "pubmed":
    from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, to_undirected)
    

    # if exists("data/" + args.datasets + ".pkl"):
    #     (data, split_edge) = torch.load("data/" + args.datasets + ".pkl")
    # else:
    from torch_geometric.data import Data
    import numpy as np
    features = np.load("new_data/node_fetures.npy")
    edge_index = np.load("new_data/edge_index.npy")
    data = Data(torch.tensor(features), torch.tensor(edge_index))
    import random
    random.seed(234)
    torch.manual_seed(234)

    data.edge_index = to_undirected(data.edge_index)
    data, split_edge = do_edge_split_weibo(data)
    torch.save((data, split_edge), "data/" + args.datasets + ".pkl")

    # import ipdb; ipdb.set_trace()
    # data.adj_t = data.edge_index

    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        if args.datasets != "collab" and args.datasets != "ppa":
            data.full_adj_t = full_edge_index
        elif args.datasets == "collab" or args.datasets == "ppa":
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    data = data.to(device)
    input_size=256

    if args.encoder == 'sage':
        model = SAGE(args.datasets, input_size, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.encoder == 'gcn':
        model = GCN(input_size, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder == 'appnp':
        model = APPNP_model(input_size, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder == 'gat':
        model = GAT(input_size, args.hidden_channels,
                    args.hidden_channels, 1,
                    args.dropout).to(device)
    elif args.encoder == 'mlp':
        model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

    # model = torch.quantization.quantize_dynamic(model, {model.convs[0].lin_l, model.convs[1].lin_l, model.convs[2].lin_l}, dtype=torch.qint8).to(device)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets == "pubmed":

    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
        'Hits@200': Logger(args.runs, args),
        'AUC': Logger(args.runs, args),
    }

    # this_file = open("this_file.csv", "a")
    val_max = 0.0
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge,
                         optimizer, args.batch_size, args.encoder, args.datasets)

            # if epoch % args.eval_steps == 0:
            results = test(model, predictor, data, split_edge,
                            evaluator, args.batch_size, args.encoder, args.datasets)

            # if not exists("saved_models/" + args.datasets + "-gcn.pkl"):
            if results['Hits@200'][1] > val_max:
                val_max = results['Hits@200'][1]
                # if args.encoder == 'sage':
                #     torch.save({'gcn': model.state_dict(), 'predictor': predictor.state_dict()}, "saved_models/" + args.datasets + "-sage.pkl")
            if results['Hits@200'][1] >= best_val:
                best_val = results['Hits@200'][1]
                cnt_wait = 0
            else:
                cnt_wait +=1

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---')

            if cnt_wait >= args.patience:
                break

    #     for key in loggers.keys():
    #         print(key)
    #         loggers[key].print_statistics(this_file,run)

    # for key in loggers.keys():
    #     print(key)
    #     loggers[key].print_statistics(this_file)


if __name__ == "__main__":
    main()
