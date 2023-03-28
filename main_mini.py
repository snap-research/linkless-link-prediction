import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import numpy as np

import torch_geometric.transforms as T
import torch_geometric

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

from utils import get_dataset, do_edge_split
from torch.nn import BCELoss

from models import MLP, GCN, SAGE, LinkPredictor, Teacher_LinkPredictor, GAT, APPNP_model
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists
from torch_cluster import random_walk
from torch.nn.functional import cosine_similarity

import time
import random
import wandb

# from torch_geometric.transforms import RandomLinkSplit

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

    return pos_batch.to("cuda"), neg_batch.to("cuda")

def train(model, predictor, t_h, teacher_predictor, data, split_edge,
                         optimizer, args, device):
    pos_train_edge = split_edge['train']['edge']
    row, col = data.adj_t

    # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets == "pubmed":
    #     row, col = data.adj_t
    # elif args.datasets == "collab":
    #     row, col,_ = data.adj_t.coo()

    # train_row, train_col = pos_train_edge.t()
    # row = torch.cat((row, train_row), dim=-1)
    # col = torch.cat((col, train_col), dim=-1) 
    # row, col = pos_train_edge.t()       

    edge_index = torch.stack([col, row], dim=0)
    # if args.KD_r:
    #     sample_step = args.rw_step
    #     pos_sample, neg_sample = neighbor_samplers(row, col, data.x, sample_step, args.ps_method, args.ns_rate)

    model.train()
    predictor.train()

    mse_loss = torch.nn.MSELoss()
    criterion = BCELoss()
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)

    total_loss = total_examples = 0

    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):
        optimizer.zero_grad()

        node_perm = next(node_loader)

        # t_h = teacher_model(data.x, data.adj_t)
        edge = pos_train_edge[link_perm].t()

        # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets== "pubmed":
        if args.datasets != "collab":
            neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0),
                                 num_neg_samples=link_perm.size(0), method='dense')
        elif args.datasets == "collab":
            neg_edge = torch.randint(0, data.x.size()[0], [edge.size(0), edge.size(1)], dtype=torch.long)
            # neg_edge = torch.randint(0, data.x.size()[0], edge.size(), dtype=torch.long, device=h.device)
        # print(kd_rank_loss)
        train_edges = torch.cat((edge, neg_edge), dim=-1).to(device)

        src = train_edges[0]
        dst = train_edges[1]

        ### calculate the kl-divergence loss 
        sample_step = args.rw_step
        pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, data.x, sample_step, args.ps_method, args.ns_rate, args.hops)

        samples = torch.cat((pos_sample, neg_sample), 1)

        this_target = torch.cat((torch.reshape(samples, (-1,)), src, dst), 0)
        h = model(data.x[this_target].to(device))

        for_loss = torch.reshape(h[:samples.size(0) * samples.size(1)], (samples.size(0), samples.size(1), args.hidden_channels))
        src_h = h[samples.size(0) * samples.size(1): samples.size(0) * samples.size(1)+src.size(0)]
        dst_h = h[samples.size(0) * samples.size(1)+src.size(0): ]

        # import ipdb; ipdb.set_trace()
        batch_emb = torch.reshape(for_loss[:,0,:], (samples[:,0].size(0), 1, h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
        t_emb = torch.reshape(t_h[samples[:,0]].to(device), (samples[:,0].size(0), 1, t_h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
        s_r = predictor(batch_emb, for_loss[:,1:,:])
        t_r = teacher_predictor(t_emb, t_h[samples[:, 1:]].to(device))
        kd_rank_loss = KD_rank(torch.reshape(s_r, (s_r.size()[0], s_r.size()[1])), torch.reshape(t_r, (t_r.size()[0], t_r.size()[1])), 1)

        rank_loss = torch.tensor(0.0).to(device)
        import itertools
        this_list = [l_i for l_i in range(sample_step*args.hops*(1+args.ns_rate))]
        # dim_pairs = [x for x in itertools.permutations(this_list, r=2)]
        dim_pairs = [x for x in itertools.combinations(this_list, r=2)]
        dim_pairs = np.array(dim_pairs).T
        teacher_rl = torch.zeros((len(t_r), dim_pairs.shape[1],1)).to(device)
                    
        mask = t_r[:, dim_pairs[0]] > (t_r[:, dim_pairs[1]] + args.margin)
        teacher_rl[mask] = 1
        mask2 = t_r[:, dim_pairs[0]] < (t_r[:, dim_pairs[1]] - args.margin)
        teacher_rl[mask2] = -1
        first_rl = s_r[:, dim_pairs[0]].squeeze()
        second_rl = s_r[:, dim_pairs[1]].squeeze()
        rank_loss = margin_rank_loss(first_rl, second_rl, teacher_rl.squeeze())
        

        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(src_h, dst_h).squeeze()
        label_loss = criterion(out, train_label)
       
        # t_out = teacher_predictor(t_h[train_edges[0]], t_h[train_edges[1]]).squeeze().detach()

        # if args.KD_kl or args.KD_r:
        #     loss = args.True_label * label_loss + args.KD_f * KD_cosine(h[node_perm], t_h[node_perm]) + args.KD_p * mse_loss(out, t_out) + args.KD_kl * kd_rank_loss + args.KD_r * rank_loss
        # else:
        #     loss = args.True_label * label_loss + args.KD_f * KD_cosine(h[node_perm], t_h[node_perm]) + args.KD_p * mse_loss(out, t_out)
        if args.KD_kl or args.KD_r:
            loss = args.True_label * label_loss + args.KD_kl * kd_rank_loss + args.KD_r * rank_loss

        # import ipdb; ipdb.set_trace()
        loss.backward()

        # for para in model.parameters():
        #     print(para.grad)
        # raise TypeError

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

    h = model(data.x.to("cuda"))

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    # if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
    if dataset != "collab":
        neg_train_edge = split_edge['train']['edge_neg'].to(h.device)
    elif dataset == "collab":
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
    if dataset != "collab":
        for K in [10, 20, 30, 50]:
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
    elif dataset == "collab":
        for K in [10, 50, 100]:
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
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--link_batch_size', type=int, default=16*1024)
    parser.add_argument('--node_batch_size', type=int, default=16*1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--datasets', type=str, default='collab')
    parser.add_argument('--predictor', type=str, default='mlp')  ##inner/mlp
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--True_label', default=0, type=float) #true_label loss
    parser.add_argument('--KD_f', default=0, type=float) #feature-based kd
    parser.add_argument('--KD_p', default=0, type=float) #predict-score kd
    parser.add_argument('--KD_kl', default=1, type=float) #kl_rank-based kd
    parser.add_argument('--KD_r', default=1, type=float) #rank-based kd
    parser.add_argument('--margin', default=0.1, type=float) #rank-based kd
    parser.add_argument('--KD_r_samp', default=10, type=int) #kl_rank-based kd
    parser.add_argument('--rw_step', type=int, default=3)
    parser.add_argument('--ns_rate', type=int, default=10)
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--ps_method', type=str, default='nb') ## positive sampling is rw or nb
    parser.add_argument('--feature_noise', type=float, default=0.0)

    args = parser.parse_args()
    print(args)
    wandb.init()

    torch_geometric.seed.seed_everything(12345)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets == "pubmed":
    if args.datasets != "collab":
        dataset = get_dataset(args.dataset_dir, args.datasets)
        data = dataset[0]

        if exists("data/" + args.datasets + ".pkl"):
            split_edge = torch.load("data/" + args.datasets + ".pkl")
        else:
            split_edge = do_edge_split(dataset)
            torch.save(split_edge, "data/" + args.datasets + ".pkl")
        
        edge_index = split_edge['train']['edge'].t()
        data.adj_t = edge_index
        input_size = data.x.size()[1]
        args.metric = 'Hits@50'
        # adj_t = split_edge['train']['edge'].t().to(device)

    elif args.datasets == "collab":
        dataset = PygLinkPropPredDataset(name='ogbl-collab')
        data = dataset[0]
        edge_index = data.edge_index
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)

        split_edge = dataset.get_edge_split()
        input_size = data.num_features
        args.metric = 'Hits@50'
        data.adj_t = edge_index

        ##add training edges into message passing
        # data.adj_t = torch.cat((edge_index, split_edge['train']['edge'].t()), dim=1)
        # split_edge['train']['edge'] = data.adj_t.t()

    args.node_batch_size = int(data.x.size()[0] / (split_edge['train']['edge'].size()[0] / args.link_batch_size))

    if 0 < args.feature_noise <= 1:
        data.x = (
            1 - args.feature_noise
        ) * data.x + args.feature_noise * torch.randn_like(data.x)

    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets == "pubmed":
        if args.datasets != "collab":
            data.full_adj_t = full_edge_index
        elif args.datasets == "collab":
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    data = data

    model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    # prepare teacher model

    # if args.encoder == 'sage':
    #     teacher_model = SAGE(args.datasets, input_size, args.hidden_channels,
    #                     args.hidden_channels, args.num_layers,
    #                     args.dropout).to(device)

    # elif args.encoder == 'gcn':
    #     teacher_model = GCN(input_size, args.hidden_channels,
    #                     args.hidden_channels, args.num_layers,
    #                     args.dropout).to(device)
        
    # elif args.encoder == 'appnp':
    #     teacher_model = APPNP_model(input_size, args.hidden_channels,
    #                 args.hidden_channels, args.num_layers,
    #                 args.dropout).to(device)
    # elif args.encoder == 'gat':
    #     teacher_model = GAT(input_size, args.hidden_channels,
    #                 args.hidden_channels, 1,
    #                 args.dropout).to(device)
                    
    pretrained_model = torch.load("saved_models/" + args.datasets + "-" + args.encoder +".pkl")

    # teacher_model.load_state_dict(pretrained_model['gcn'], strict=True)
    teacher_predictor = Teacher_LinkPredictor(args.predictor, 256, 256, 1,
                             2, args.dropout)
    teacher_predictor.load_state_dict(pretrained_model['predictor'], strict=True)
    # teacher_model.to(device)
    teacher_predictor.to(device)

    # h = teacher_model(data.x.to(device), data.adj_t.to(device))

    # t_h = teacher_model(data.x.to(device), data.adj_t.to(device))
    # torch.save({'features': t_h}, "saved_features-collab.pkl")
    # raise TypeError
    t_h = torch.load("./Saved_features/" + args.datasets + "_transductive.pkl")
    t_h = t_h['features']

    # losses_weights = torch.nn.Parameter(torch.tensor(np.ones(4)))

    # for para in teacher_model.parameters():
    #     para.requires_grad=False
    for para in teacher_predictor.parameters():
        para.requires_grad=False

    evaluator = Evaluator(name='ogbl-ddi')
    # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets == "pubmed":
    if args.datasets != "collab":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@20': Logger(args.runs, args),
            'Hits@30': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }
    elif args.datasets == "collab":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }

    val_max = 0.0
    this_file = open("results/" + args.datasets + "_" + str(args.True_label) + "_" + str(args.KD_f) + "_" + str(args.KD_p) + "_" + str(args.KD_kl) + "_" + str(args.KD_r) + "_step" + str(args.rw_step) + "_" + args.ps_method + "_rate" + str(args.ns_rate) + "\n" +  ".csv", "a")

    for run in range(args.runs):
        best_test = 0.0

        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, t_h, teacher_predictor, data, split_edge,
                         optimizer, args, device)

            # if epoch % args.eval_steps == 0:
            results = test(model, predictor, data, split_edge,
                            evaluator, args.link_batch_size, args.encoder, args.datasets)

            if results[args.metric][1] >= best_val:
                best_val = results[args.metric][1]
                best_test = results[args.metric][2]
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
                    this_file.write(f'Run: {run + 1:02d}, ' + str(epoch+1) + f'Loss: {loss:.4f}, ' + key + ": " +  f'Train: {100 * train_hits:.2f}%, ' + "," + f'Valid: {100 * valid_hits:.2f}%, ' + "," + f'Test: {100 * test_hits:.2f}%' + "\n" )

                print('---')

            if cnt_wait >= args.patience:
                break
            metrics = {args.metric: results[args.metric][1], 'test_result': best_test, 'loss': loss}
            wandb.log(metrics)

            
        for key in loggers.keys():
            print(key)
            this_file.write(str(run) + "," + key + ": ")
            loggers[key].print_statistics(this_file, run)

    for key in loggers.keys():
        print(key)
        this_file.write("All,  " + key + ": ")
        loggers[key].print_statistics(this_file)

        if key == "Hits@50":
            best_results = []
            for r in loggers[key].results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test1 = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test1))

            best_result = torch.tensor(best_results)

            this_file.write(f'All runs:   \n')
            r = best_result[:, 0]
            this_file.write(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            this_file.write(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            this_file.write(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            this_file.write(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            this_file.write("\n")
        metrics["Mean_"+key] = r.mean()
        metrics["Mean_" + key + "_std"] = r.std()
        wandb.log(metrics)

    this_file.close()


if __name__ == "__main__":
    main()
