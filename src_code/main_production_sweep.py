import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import numpy as np

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger_production import Logger

from utils import get_dataset, do_edge_split
from torch.nn import BCELoss

from models import MLP, GCN, SAGE, LinkPredictor, Teacher_LinkPredictor
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists
from torch_cluster import random_walk
from torch.nn.functional import cosine_similarity

import time
import random

import wandb

def KD_cosine(s, t):
    return 1-cosine_similarity(s, t.detach(), dim=-1).mean()

def KD_rank(s,t,T):
    y_s = F.log_softmax(s/T, dim=-1)
    y_t = F.softmax(t/T, dim=-1)
    loss = F.kl_div(y_s,y_t,size_average=False) * (T**2) / y_s.size()[0]
    return loss

def neighbor_samplers(row, col, sample, x, step, ps_method, ns_rate, hops):
    batch = sample

    if ps_method == 'rw':
        pos_batch = random_walk(row, col, batch, walk_length=step*hops,
                                coalesced=False)
    elif ps_method == 'nb':
        pos_batch = None
        for i in range(step):
            if pos_batch is None:
                pos_batch = random_walk(row, col, batch, walk_length=hops,
                                    coalesced=False)
            else:
                # print(torch.reshape(random_walk(row, col, batch, walk_length=1,coalesced=False)[:,1], (-1,1)))
                pos_batch = torch.cat((pos_batch, random_walk(row, col, batch, walk_length=hops,coalesced=False)[:,1:]), 1)

    neg_batch = torch.randint(0, x.size(0), (batch.numel(), step*hops*ns_rate),
                                  dtype=torch.long)

    return pos_batch.to(x.device), neg_batch.to(x.device)

def train(model, predictor, t_h, teacher_model, teacher_predictor, training_data,
                         optimizer, args):
    pos_train_edge = training_data.edge_index.t()
    row, col = training_data.edge_index

    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    mse_loss = torch.nn.MSELoss()
    criterion = BCELoss()
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)

    total_loss = total_examples = 0

    node_loader = iter(DataLoader(range(training_data.x.size(0)), args.node_batch_size, shuffle=True))
    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):
        optimizer.zero_grad()

        node_perm = next(node_loader).to(training_data.x.device)

        h = model(training_data.x)
        # t_h = teacher_model(training_data.x, training_data.edge_index)
        edge = pos_train_edge[link_perm].t()

        if args.KD_r or args.KD_kl:
            sample_step = args.rw_step
            pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, training_data.x, sample_step, args.ps_method, args.ns_rate, args.hops)

            ### calculate the distribution based matching loss 
            samples = torch.cat((pos_sample, neg_sample), 1)
            batch_emb = torch.reshape(h[samples[:,0]], (samples[:,0].size(0), 1, h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
            t_emb = torch.reshape(t_h[samples[:,0]], (samples[:,0].size(0), 1, h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
            s_r = predictor(batch_emb, h[samples[:, 1:]])
            t_r = teacher_predictor(t_emb, t_h[samples[:, 1:]])
            kd_rank_loss = KD_rank(torch.reshape(s_r, (s_r.size()[0], s_r.size()[1])), torch.reshape(t_r, (t_r.size()[0], t_r.size()[1])), 1)

            ### calculate the rank based matching loss
            rank_loss = torch.tensor(0.0).to("cuda")
            import itertools
            this_list = [l_i for l_i in range(sample_step*args.hops*(1+args.ns_rate))]
            dim_pairs = [x for x in itertools.combinations(this_list, r=2)]
            dim_pairs = np.array(dim_pairs).T
            teacher_rl = torch.zeros((len(t_r), dim_pairs.shape[1],1)).to(t_r.device)
                      
            mask = t_r[:, dim_pairs[0]] > (t_r[:, dim_pairs[1]] + args.margin)
            teacher_rl[mask] = 1
            mask2 = t_r[:, dim_pairs[0]] < (t_r[:, dim_pairs[1]] - args.margin)
            teacher_rl[mask2] = -1
            first_rl = s_r[:, dim_pairs[0]].squeeze()
            second_rl = s_r[:, dim_pairs[1]].squeeze()
            rank_loss = margin_rank_loss(first_rl, second_rl, teacher_rl.squeeze())
       
        neg_edge = negative_sampling(edge_index, num_nodes=training_data.x.size(0),
                                num_neg_samples=link_perm.size(0), method='dense')
        
        ### True_label loss  
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        label_loss = criterion(out, train_label)
       
        t_out = teacher_predictor(t_h[train_edges[0]], t_h[train_edges[1]]).squeeze().detach()

        if args.KD_kl or args.KD_r:
            loss = args.True_label * label_loss + args.KD_f * KD_cosine(h[node_perm], t_h[node_perm]) + args.KD_p * mse_loss(out, t_out) + args.KD_kl * kd_rank_loss + args.KD_r * rank_loss
        else:
            loss = args.True_label * label_loss + args.KD_f * KD_cosine(h[node_perm], t_h[node_perm]) + args.KD_p * mse_loss(out, t_out)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(training_data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        
    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, val_data, inference_data, test_edge_bundle, negative_samples, evaluator, batch_size, encoder_name, dataset):
    model.eval()
    predictor.eval()

    h = model(val_data.x)
  
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

    h = model(inference_data.x)
   
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

    return results

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--link_batch_size', type=int, default=64*1024)
    parser.add_argument('--node_batch_size', type=int, default=64*1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--datasets', type=str, default='cora')
    parser.add_argument('--predictor', type=str, default='mlp')  ##inner/mlp
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@50', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--True_label', default=1000, type=float) #true_label loss
    parser.add_argument('--KD_f', default=0, type=float) #Representation-based matching KD
    parser.add_argument('--KD_p', default=0, type=float) #logit-based matching KD
    parser.add_argument('--KD_kl', default=0.1, type=float) #distribution-based matching kd
    parser.add_argument('--KD_r', default=100, type=float) #rank-based matching kd
    parser.add_argument('--margin', default=0.1, type=float) #margin for rank-based kd
    parser.add_argument('--rw_step', type=int, default=3) # nearby nodes sampled times
    parser.add_argument('--ns_rate', type=int, default=5) # # randomly sampled rate over # nearby nodes
    parser.add_argument('--hops', type=int, default=2) # random_walk step for each sampling time
    parser.add_argument('--ps_method', type=str, default='nb') # positive sampling is rw or nb

    args = parser.parse_args()
    print(args)

    wandb.init()

    # this_file = "../results/" + args.datasets + "_KD_production.txt"
    # file = open(this_file, "a")
    # file.write(str(args)+"\n")
    # if args.KD_f != 0:
    #     file.write("Logit-matching\n")
    # elif args.KD_p != 0:
    #     file.write("Representation-matching\n")
    # elif args.KD_kl != 0 or args.KD_r != 0:
    #     file.write("LLP (Relational Distillation)\n")
    # file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #### prepare the datasets
    training_data, val_data, inference_data, data, test_edge_bundle, negative_samples = torch.load("../data/" + args.datasets + "_production.pkl")
        
    input_size = training_data.x.size(1)

    training_data.to(device)
    val_data.to(device)
    inference_data.to(device)

    args.node_batch_size = int(training_data.x.size()[0] / (training_data.edge_index.size(1) / args.link_batch_size))

    ### prepare the student and teacher models
    model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    # prepare teacher model
    teacher_model = SAGE(args.datasets, input_size, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    
   
    pretrained_model = torch.load("../saved_models/" + args.datasets + "-sage_production.pkl")
    # teacher_model.load_state_dict(pretrained_model['gnn'], strict=True)
    teacher_predictor = Teacher_LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)
    teacher_predictor.load_state_dict(pretrained_model['predictor'], strict=True)
    # teacher_model.to(device)
    teacher_predictor.to(device)

    t_h = torch.load("../Saved_features/" + args.datasets + "-sage_production.pkl")
    t_h = t_h['features'].to(device)

    for para in teacher_model.parameters():
        para.requires_grad=False
    for para in teacher_predictor.parameters():
        para.requires_grad=False

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'AUC': Logger(args.runs, args),
    }

    val_max = 0.0
    for run in range(args.runs):
        import torch_geometric
        torch_geometric.seed.seed_everything(run+1)

        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        best_test = 0.0
        best_new_new=0.0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, t_h, teacher_model, teacher_predictor, training_data,
                         optimizer, args)

            results = test(model, predictor, val_data, inference_data, test_edge_bundle, negative_samples,
                        evaluator, args.link_batch_size, args.encoder, args.datasets)
          
            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                best_test = results["Hits@20"][1]
                best_new_new = results["Hits@20"][4]
                cnt_wait = 0
            else:
                cnt_wait +=1

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
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

            metrics = {"Hits@20": results["Hits@20"][0], 'test_result': results["Hits@20"][1], 'loss': loss, 'best_test': best_test, 'best_new_new': best_new_new}
            wandb.log(metrics)

            if cnt_wait >= args.patience:
                break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    # file = open(this_file, "a")
    # file.write(f'All runs:\n')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

        # file.write(f'{key}:\n')
        if key == "Hits@20":
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

            r = best_result[:, 1]
            metrics["Mean_test_Hits@20"] = r.mean()
            metrics["Mean_test_std"] = r.std()            
            r = best_result[:, 2]
            metrics["Mean_old-old_Hits@20"] = r.mean()
            metrics["Mean_old-old_std"] = r.std()
            r = best_result[:, 3]
            metrics["Mean_old-old_Hits@20"] = r.mean()
            metrics["Mean_old-old_std"] = r.std()
            r = best_result[:, 4]
            metrics["Mean_new-new_Hits@20"] = r.mean()
            metrics["Mean_new-new_std"] = r.std()
            
            wandb.log(metrics)

if __name__ == "__main__":
    main()