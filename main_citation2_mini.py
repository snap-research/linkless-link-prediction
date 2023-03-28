import argparse

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from torch_geometric.utils import negative_sampling
import numpy as np

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from citation_logger import Logger

from utils import get_dataset, do_edge_split
from torch.nn import BCELoss

from models import MLP, GCN, SAGE, LinkPredictor, Teacher_LinkPredictor, GAT, APPNP_model
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists
from torch_cluster import random_walk
from torch.nn.functional import cosine_similarity
import torch_geometric

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

def neighbor_samplers(row, col, sample, x, step, ps_method, ns_rate, hops, device):
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

    return pos_batch.to(device), neg_batch.to(device)

def train(model, predictor, t_h, teacher_predictor, data, split_edge, optimizer, batch_size, args, device):
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node']
    target_edge = split_edge['train']['target_node']

    mse_loss = torch.nn.MSELoss()
    criterion = BCELoss()
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)

    total_loss = total_examples = 0
    row, col, _ = data.adj_t.coo()

    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    for perm in DataLoader(range(source_edge.size(0)), args.link_batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        node_perm = next(node_loader)

        # t_h = teacher_model(data.x, data.adj_t)

        src, dst = source_edge[perm].to(device), target_edge[perm].to(device)

        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long).to(device)

        sample_step = args.rw_step
        pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, data.x, sample_step, args.ps_method, args.ns_rate, args.hops, device)

        ### calculate the kl-divergence loss 
        samples = torch.cat((pos_sample, neg_sample), 1)

        this_target = torch.cat((torch.reshape(samples, (-1,)), src, dst, dst_neg), 0)
        h = model(data.x[this_target].to(device))

        for_loss = torch.reshape(h[:samples.size(0) * samples.size(1)], (samples.size(0), samples.size(1), args.hidden_channels))
        src_h = h[samples.size(0) * samples.size(1): samples.size(0) * samples.size(1)+src.size(0)]
        dst_h = h[samples.size(0) * samples.size(1)+src.size(0): samples.size(0) * samples.size(1)+src.size(0)+dst.size(0)]
        dst_neg_h = h[samples.size(0) * samples.size(1)+src.size(0)+dst.size(0):]

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
        

        pos_out = predictor(src_h, dst_h)
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        
        neg_out = predictor(src_h, dst_neg_h)
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        train_out = torch.cat((pos_out, neg_out), dim=0).squeeze()
        train_label = torch.cat((torch.ones(pos_out.size()[0]), torch.zeros(neg_out.size()[0])), dim=0).to(h.device)
        label_loss = criterion(train_out, train_label)


        # t_out = torch.cat((teacher_predictor(t_h[src], t_h[dst]), teacher_predictor(t_h[src], t_h[dst_neg])), dim=0).squeeze().detach()

        # if args.KD_kl or args.KD_r:
        #     loss = args.True_label * label_loss + args.KD_f * KD_cosine(h[node_perm], t_h[node_perm]) + args.KD_p * mse_loss(train_out, t_out) + args.KD_kl * kd_rank_loss + args.KD_r * rank_loss
        # else:
        #     loss = args.True_label * label_loss + args.KD_f * KD_cosine(h[node_perm], t_h[node_perm]) + args.KD_p * mse_loss(train_out, t_out)
        if args.KD_kl or args.KD_r:
            loss = args.True_label * label_loss + args.KD_kl * kd_rank_loss + args.KD_r * rank_loss
        # loss = pos_loss + neg_loss
        # loss = label_loss
        # import ipdb; ipdb.set_trace()
        loss.backward()
        optimizer.step()
        print(loss)

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    predictor.eval()

    def test_split(split):
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        target_neg = split_edge[split]['target_node_neg']
        
        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(model(data.x[src].to("cuda")), model(data.x[dst].to("cuda"))).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(model(data.x[src].to("cuda")), model(data.x[dst_neg].to("cuda"))).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    # def test_split(split):
    #     import ipdb; ipdb.set_trace()
    #     source = split_edge[split]['source_node'].to(h.device)
    #     target = split_edge[split]['target_node'].to(h.device)
    #     target_neg = split_edge[split]['target_node_neg'].to(h.device)

    #     this_target = torch.concat((source, target, torch.reshape((target_neg.shape[0] * target_neg.shape[1]),0)), 0)
    #     h = model(data.x[this_target].to("cuda"))
        
    #     pos_preds = []
    #     source_h = h[:source.shape[0]]
    #     target_h = h[source.shape[0]:source.shape[0]+target.shape[0]]
    #     target_neg_h = h[source.shape[0]+target.shape[0]:]


    #     for perm in DataLoader(range(source.size(0)), batch_size):
    #         # src, dst = source[perm], target[perm]
    #         pos_preds += [predictor(source_h[perm], target_h[perm]).squeeze().cpu()]
    #     pos_pred = torch.cat(pos_preds, dim=0)

    #     neg_preds = []
    #     source = source.view(-1, 1).repeat(1, 1000).view(-1)
    #     target_neg = target_neg.view(-1)
    #     for perm in DataLoader(range(source.size(0)), batch_size):
    #         src, dst_neg = source[perm], target_neg[perm]
    #         neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
    #     neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

    #     return evaluator.eval({
    #         'y_pred_pos': pos_pred,
    #         'y_pred_neg': neg_pred,
    #     })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--link_batch_size', type=int, default=16*1024)
    parser.add_argument('--node_batch_size', type=int, default=16*1024)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--True_label', default=0, type=float) #true_label loss
    parser.add_argument('--KD_f', default=0, type=float) #feature-based kd
    parser.add_argument('--KD_p', default=0, type=float) #predict-score kd
    parser.add_argument('--KD_kl', default=1, type=float) #kl_rank-based kd
    parser.add_argument('--KD_r', default=1, type=float) #rank-based kd
    parser.add_argument('--margin', default=0.1, type=float) #rank-based kd
    parser.add_argument('--KD_r_samp', default=10, type=int) #kl_rank-based kd
    parser.add_argument('--rw_step', type=int, default=1)
    parser.add_argument('--ns_rate', type=int, default=3)
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--ps_method', type=str, default='nb') ## positive sampling is rw or nb

    args = parser.parse_args()
    print(args)
    torch_geometric.seed.seed_everything(12345)
    wandb.init()

    gpu = 0
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    # device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data

    split_edge = dataset.get_edge_split()

    args.node_batch_size = int(data.x.size(0) / (split_edge['train']['source_node'].size(0) / args.link_batch_size))

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }


    model = MLP(args.num_layers, data.num_features, args.hidden_channels, args.hidden_channels, args.dropout).to(device)
    pretrained_model = torch.load("saved_models/citation2-sage.pkl")

    # teacher_model = SAGE("citation2", data.num_features, 256,
    #                 256, 3,
    #                 0.).to(device)
    # teacher_model.load_state_dict(pretrained_model['gcn'], strict=True)
    teacher_predictor = Teacher_LinkPredictor("mlp", 256, 256, 1,
                              3, 0.).to(device)
    teacher_predictor.load_state_dict(pretrained_model['predictor'], strict=True)
    # teacher_model.to(device)
    teacher_predictor.to(device)

    t_h = torch.load("./Saved_features/citation2.pkl")
    t_h = t_h['features']

    # for para in teacher_model.parameters():
    #     para.requires_grad=False
    for para in teacher_predictor.parameters():
        para.requires_grad=False

    # else:
    #     raise TypeError
    #     model = GCN(data.num_features, args.hidden_channels,
    #                 args.hidden_channels, args.num_layers,
    #                 args.dropout).to(device)

    #     # Pre-compute GCN normalization.
    #     adj_t = data.adj_t.set_diag()
    #     deg = adj_t.sum(dim=1).to(torch.float)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    #     adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    #     data.adj_t = adj_t

    predictor = LinkPredictor("mlp", args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-citation2')
    logger = Logger(args.runs, args)

    val_max = 0.0
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, t_h, teacher_predictor, data, split_edge, optimizer,
                         args.batch_size, args, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                result = test(model, predictor, data, split_edge, evaluator,
                              args.batch_size)
                logger.add_result(run, result)
                train_mrr, valid_mrr, test_mrr = result

                metrics = {"train": train_mrr, "train": valid_mrr, 'test_result': test_mrr, 'loss': loss}
                wandb.log(metrics)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')
                # if val_max < valid_mrr:
                #     val_max = valid_mrr
                #     torch.save({'gcn': model.state_dict(), 'predictor': predictor.state_dict()}, "saved_models/citation2-sage.pkl")


        print('GraphSAGE' if args.use_sage else 'GCN')
        logger.print_statistics(run)
    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics()


if __name__ == "__main__":
    main()