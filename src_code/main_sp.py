import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

from utils import get_dataset, do_edge_split
from torch.nn import BCELoss

from models import MLP, GCN, SAGE, LinkPredictor, GAT, APPNP_model
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists

def train(model, predictor, data, split_edge, optimizer, batch_size, encoder_name, dataset):

    row, col = data.adj_t
    
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    criterion = BCELoss()
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

        if dataset != "collab":
            neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')
        elif dataset == "collab":
            neg_edge = torch.randint(0, data.x.size()[0], edge.size(), dtype=torch.long,
                             device=h.device)

        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
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

    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

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
    if dataset != "collab":
        for K in [10, 20, 30, 50]:
            evaluator.K = K
 
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (valid_hits, test_hits)
    elif dataset == "collab":
        for K in [10, 50, 100]:
            evaluator.K = K
 
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (valid_hits, test_hits)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()),roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()))

    return results, h

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--datasets', type=str, default='cora')
    parser.add_argument('--predictor', type=str, default='mlp')  ##inner/mlp
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')

    args = parser.parse_args()
    print(args)

    this_file = "../results/" + args.datasets + "_supervised_transductive_search.txt"
    file = open(this_file, "a")
    file.write(str(args))
    file.write(args.encoder + " as the encoder\n")
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.datasets != "collab":
        dataset = get_dataset(args.dataset_dir, args.datasets)
        data = dataset[0]

        if exists("../data/" + args.datasets + ".pkl"):
            split_edge = torch.load("../data/" + args.datasets + ".pkl")
        else:
            split_edge = do_edge_split(dataset)
            torch.save(split_edge, "../data/" + args.datasets + ".pkl")
        
        edge_index = split_edge['train']['edge'].t()
        data.adj_t = edge_index
        input_size = data.x.size()[1]
        args.metric = 'Hits@20'

    elif args.datasets == "collab":
        dataset = PygLinkPropPredDataset(name=('ogbl-' + args.datasets))
        data = dataset[0]
        edge_index = data.edge_index
        data = T.ToSparseTensor()(data)

        split_edge = dataset.get_edge_split()
        input_size = data.num_features
        args.metric = 'Hits@50'

        data.adj_t = edge_index
    
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

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    if args.datasets != "collab" and args.datasets != "ppa":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@20': Logger(args.runs, args),
            'Hits@30': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }
    elif args.datasets == "collab" or args.datasets == "ppa":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }

    val_max = 0.0
    for run in range(args.runs):
        import torch_geometric
        torch_geometric.seed.seed_everything(run)
        
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

            results, h = test(model, predictor, data, split_edge,
                            evaluator, args.batch_size, args.encoder, args.datasets)

            if results['Hits@50'][0] > val_max:
                val_max = results['Hits@50'][0]
                if args.encoder == 'sage':
                    torch.save({'features': h}, "../Saved_features/" + args.datasets + "-" + args.encoder + "_transductive.pkl")
                    torch.save({'gnn': model.state_dict(), 'predictor': predictor.state_dict()}, "../saved_models/" + args.datasets + "-" + args.encoder + "_transductive.pkl")
            if results['Hits@50'][0] >= best_val:
                best_val = results['Hits@50'][0]
                cnt_wait = 0
            else:
                cnt_wait +=1

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---')

            if cnt_wait >= args.patience:
                break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    file = open(this_file, "a")
    file.write(f'All runs:\n')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        
        file.write(f'{key}:\n')
        best_results = []
        for r in loggers[key].results:
            r = 100 * torch.tensor(r)
            valid = r[:, 0].max().item()
            test1 = r[r[:, 0].argmax(), 1].item()
            best_results.append((valid, test1))

        best_result = torch.tensor(best_results)

        r = best_result[:, 1]
        file.write(f'Test: {r.mean():.4f} ± {r.std():.4f}\n')
    file.close()


if __name__ == "__main__":
    main()
