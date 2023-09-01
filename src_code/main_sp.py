import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger, Logger_production

from utils import get_dataset, do_edge_split

from models import MLP, GCN, SAGE, LinkPredictor
from sageconv_updated import SAGEConv_update
from torch_geometric.nn import SAGEConv
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists

def train(model, predictor, data, split_edge, optimizer, batch_size, encoder_name, dataset, transductive):
    if transductive == "transductive":
        row, col = data.adj_t
        pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    else:
        row, col = data.edge_index
        pos_train_edge = data.edge_index.t()
        
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    bce_loss = nn.BCELoss()


    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        if encoder_name == 'mlp':
            h = model(data.x)
        else:
            if transductive == "transductive":
                h = model(data.x, data.adj_t)
            else:
                h = model(data.x, data.edge_index)

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
        loss = bce_loss(out, train_label)

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
def test_transductive(model, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset):
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

@torch.no_grad()
def test_production(model, predictor, val_data, inference_data, test_edge_bundle, negative_samples, evaluator, batch_size, encoder_name, dataset):
    model.eval()
    predictor.eval()

    if encoder_name == 'mlp':
        h = model(val_data.x)
    else:
        h = model(val_data.x, val_data.edge_index)

    saved_h = h

    negative_edges = negative_samples.t().to(h.device)
    val_edges = val_data.edge_label_index.t()
    val_pos_edges = val_edges[val_data.edge_label.bool()]
    val_neg_edges = val_edges[(torch.tensor(1)-val_data.edge_label).bool()] 
    old_old_edges = test_edge_bundle[0].t().to(h.device)
    old_new_edges = test_edge_bundle[1].t().to(h.device)
    new_new_edges = test_edge_bundle[2].t().to(h.device)
    test_edges = test_edge_bundle[3].t().to(h.device)

    pos_valid_preds = []
    for perm in DataLoader(range(val_pos_edges.size(0)), batch_size):
        edge = val_pos_edges[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_preds = []
    for perm in DataLoader(range(val_neg_edges.size(0)), batch_size):
        edge = val_neg_edges[perm].t()
        neg_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_pred = torch.cat(neg_preds, dim=0)

    if encoder_name == 'mlp':
        h = model(inference_data.x)
    else:
        h = model(inference_data.x, inference_data.edge_index)

    pos_test_preds = []
    for perm in DataLoader(range(test_edges.size(0)), batch_size):
        edge = test_edges[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    old_old_pos_test_preds = []
    for perm in DataLoader(range(old_old_edges.size(0)), batch_size):
        edge = old_old_edges[perm].t()
        old_old_pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    old_old_pos_test_pred = torch.cat(old_old_pos_test_preds, dim=0)

    old_new_pos_test_preds = []
    for perm in DataLoader(range(old_new_edges.size(0)), batch_size):
        edge = old_new_edges[perm].t()
        old_new_pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    old_new_pos_test_pred = torch.cat(old_new_pos_test_preds, dim=0)

    new_new_pos_test_preds = []
    for perm in DataLoader(range(new_new_edges.size(0)), batch_size):
        edge = new_new_edges[perm].t()
        new_new_pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    new_new_pos_test_pred = torch.cat(new_new_pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(negative_edges.size(0)), batch_size):
        edge = negative_edges[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30, 50]:
        evaluator.K = K
        val_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        old_old_test_hits = evaluator.eval({
            'y_pred_pos': old_old_pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        old_new_test_hits = evaluator.eval({
            'y_pred_pos': old_new_pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        new_new_test_hits = evaluator.eval({
            'y_pred_pos': new_new_pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (val_hits, test_hits, old_old_test_hits, old_new_test_hits, new_new_test_hits)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    old_old_result = torch.cat((torch.ones(old_old_pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    old_old_pred = torch.cat((old_old_pos_test_pred, neg_test_pred), dim=0)

    old_new_result = torch.cat((torch.ones(old_new_pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    old_new_pred = torch.cat((old_new_pos_test_pred, neg_test_pred), dim=0)

    new_new_result = torch.cat((torch.ones(new_new_pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    new_new_pred = torch.cat((new_new_pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()), roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()),roc_auc_score(old_old_result.cpu().numpy(),old_old_pred.cpu().numpy()),roc_auc_score(old_new_result.cpu().numpy(),old_new_pred.cpu().numpy()),roc_auc_score(new_new_result.cpu().numpy(),new_new_pred.cpu().numpy()))

    return results, saved_h

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
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner','mlp'])
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--transductive', type=str, default='transductive', choices=['transductive', 'production'])
    parser.add_argument('--minibatch', action='store_true')


    args = parser.parse_args()
    print(args)

    Logger_file = "../results/" + args.datasets + "_supervised_" + args.transductive + ".txt"
    file = open(Logger_file, "a")
    file.write(str(args))
    file.write(args.encoder + " as the encoder\n")
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.transductive == "transductive": 
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

    else:
        training_data, val_data, inference_data, _, test_edge_bundle, negative_samples = torch.load("../data/" + args.datasets + "_production.pkl")
        input_size = training_data.x.size(1)
        
        args.metric = 'Hits@20'

        training_data.to(device)
        val_data.to(device)
        inference_data.to(device)

    if args.encoder == 'sage':
        if args.datasets == "coauthor-physics":
            model = SAGE(args.datasets, input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout, SAGEConv_update).to(device)
        else:
            model = SAGE(args.datasets, input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout, SAGEConv).to(device)
    elif args.encoder == 'gcn':
        model = GCN(input_size, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder == 'mlp':
        model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    if args.transductive == "transductive":
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
    else:
        loggers = {
            'Hits@10': ProductionLogger(args.runs, args),
            'Hits@20': ProductionLogger(args.runs, args),
            'Hits@30': ProductionLogger(args.runs, args),
            'Hits@50': ProductionLogger(args.runs, args),
            'AUC': ProductionLogger(args.runs, args),
        }

    val_max = 0.0
    for run in range(args.runs):
        torch_geometric.seed.seed_everything(run)
        
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        for epoch in range(1, 1 + args.epochs):
            if args.transductive == "transductive":
                loss = train(model, predictor, data, split_edge,
                            optimizer, args.batch_size, args.encoder, args.datasets, args.transductive)
                
                results, h = test_transductive(model, predictor, data, split_edge,
                            evaluator, args.batch_size, args.encoder, args.datasets)
            else:
                loss = train(model, predictor, training_data, None,
                         optimizer, args.batch_size, args.encoder, args.datasets, args.transductive)
                
                results, h = test_production(model, predictor, val_data, inference_data, test_edge_bundle, negative_samples,
                            evaluator, args.batch_size, args.encoder, args.datasets)

            if results[args.metric][0] > val_max:
                val_max = results[args.metric][0]
                if args.encoder == 'sage':
                    torch.save({'features': h}, "../Saved_features/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl")
                    torch.save({'gnn': model.state_dict(), 'predictor': predictor.state_dict()}, "../saved_models/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl")
            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait +=1

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                if args.transductive:
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                else:
                    for key, result in results.items():
                        valid_hits, test_hits, old_old, old_new, new_new = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'valid: {100 * valid_hits:.2f}%, '
                            f'test: {100 * test_hits:.2f}%, '
                            f'old_old: {100 * old_old:.2f}%, '
                            f'old_new: {100 * old_new:.2f}%, '
                            f'new_new: {100 * new_new:.2f}%')
                print('---')

            if cnt_wait >= args.patience:
                break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    file = open(Logger_file, "a")
    file.write(f'All runs:\n')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        
        if args.transductive:
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
        else:
            file.write(f'{key}:\n')
            best_results = []
            for r in loggers[key].results:
                r = 100 * torch.tensor(r)
                val = r[r[:, 0].argmax(), 0].item()
                test_r = r[r[:, 0].argmax(), 1].item()
                old_old = r[r[:, 0].argmax(), 2].item()
                old_new = r[r[:, 0].argmax(), 3].item()
                new_new = r[r[:, 0].argmax(), 4].item()
                best_results.append((val, test_r, old_old, old_new, new_new))

            best_result = torch.tensor(best_results)

            r = best_result[:, 0]
            file.write(f'  Final val: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            file.write(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            file.write(f'   Final old_old: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            file.write(f'   Final old_new: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            file.write(f'   Final new_new: {r.mean():.2f} ± {r.std():.2f}\n')
    file.close()


main()
