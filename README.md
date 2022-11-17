# Linkless Link Prediction via Relational Distillation

## Introduction
This repository contains the source code for the paper: Linkless Link Prediction via Relational Distillation
<!-- This repository contains the source code for the paper: [Linkless Link Prediction via Relational Distillation](https://arxiv.org/pdf/2210.05801.pdf)  -->
<img width="1209" alt="image" src="https://user-images.githubusercontent.com/69767476/193711518-fdc8c163-7bbc-4118-ad55-75835954d2c7.png">

## Requirements
```
- torch==1.12.1
- torch-geometric==2.1.0
- numpy
- ogb==1.3.5
- sklearn==1.1.2
```

## Preprocess the dataset for the production setting
- Change the dataset name ("dataset") in Line 280 in inductive_splitting_node.py file. 
```
python inductive_splitting_node.py
```

## Run LLP
### Transductive Setting
- **Teacher GNN training.** We can change "sage" to "mlp" to obtain supervised training results with MLP.
```
python main_sp.py --datasets=cora --encoder=sage 
```
- **Student MLP training.** KD_kl and KD_r indicate the weights for the distribution-based and rank-based matching KD, respectively.
```
python main.py --datasets=cora --KD_kl=1 --KD_r=1 --True_label=1
```
### Production Setting
- **Teacher GNN training.** We can change "sage" to "mlp" to obtain the supervised training results with MLP.
```
python main_sp_production.py --datasets=cora --encoder=sage 
```
- **Student MLP training.** KD_kl and KD_r indicate the weights for the distribution-based and rank-based matching KD, respectively.
```
python main_production.py --datasets=cora --KD_kl=1 --KD_r=1 --True_label=1
```
## Citation

## Acknowledgements


<!-- by Zhichun Guo(zguo5@nd.edu), William Shiao(wshiao@snap.com), Shichang Zhang(shichang@cs.ucla.edu), Yozen Liu(yliu2@snapchat.com), Nitesh Chawla(nchawla@nd.edu), Neil Shah(nshah@snap.com), Tong Zhao(tzhao@snapchat.com).
 -->
