# Linkless Link Prediction via Relational Distillation

## Introduction
This repository contains the source code for the paper: [Linkless Link Prediction via Relational Distillation](https://arxiv.org/pdf/2210.05801.pdf) 
<img width="1209" alt="image" src="https://user-images.githubusercontent.com/69767476/193711518-fdc8c163-7bbc-4118-ad55-75835954d2c7.png">

## Requirements
```
- torch==1.12.1
- torch-geometric==2.1.0
- numpy
- ogb==1.3.5
- sklearn==1.1.2
- python==3.9.13
```

<!-- ## Preprocess the dataset for the production setting
- Change the dataset name ("dataset") in Line 205 in production_splitting_node.py file. 
```
python production_splitting_node.py
``` -->

## Usage
### Transductive Setting 
- **Teacher GNN training.** You can change "sage" to "mlp" to obtain supervised training results with MLP. 
```
python main_sp.py --datasets=cora --encoder=sage 
```
To reproduce the supervised results shown in Table 2, you can just simply run the following command. The results will be shown in /results.
```
cd scripts/
bash supervised_transductive.sh
```
- **Student MLP training.** KD_kl and KD_r indicate the weights for the distribution-based and rank-based matching KD, respectively.
```
python main.py --datasets=cora --KD_kl=1 --KD_r=1 --True_label=1
```
To reproduce the results shown in Table 2, please run the following command:
```
cd scripts/
bash KD_transductive.sh
```
### Production Setting
- **Pre-process dataset**
In this work, we design a new production setting to resemble the real-world link prediction scenario. For more details, please refer to our paper. Our split datasets are already saved in ../data folder. If you want to apply this setting on our own datasets or split the datsets by your self, please change the dataset name ("dataset") in Line 205 in production_splitting_node.py file and run the following command:
```
python production_splitting_node.py
```
- **Teacher GNN training.** We can change "sage" to "mlp" to obtain the supervised training results with MLP.
```
python main_sp_production.py --datasets=cora --encoder=sage 
```
To reproduce the supervised results shown in Table 3, you can just simply run the following command. The results will be shown in /results.
```
cd scripts/
bash supervised_production.sh
```
- **Student MLP training.** KD_kl and KD_r indicate the weights for the distribution-based and rank-based matching KD, respectively.
```
python main_production.py --datasets=cora --KD_kl=1 --KD_r=1 --True_label=1
```
To reproduce the results shown in Table 3, please run the following command:
```
cd scripts/
bash KD_production.sh
```
## Reference
```
@article{guo2022linkless,
  title={Linkless Link Prediction via Relational Distillation},
  author={Guo, Zhichun and Shiao, William and Zhang, Shichang and Liu, Yozen and Chawla, Nitesh and Shah, Neil and Zhao, Tong},
  journal={arXiv preprint arXiv:2210.05801},
  year={2022}
}
```

<!-- ## Acknowledgements -->


<!-- by Zhichun Guo(zguo5@nd.edu), William Shiao(wshiao@snap.com), Shichang Zhang(shichang@cs.ucla.edu), Yozen Liu(yliu2@snapchat.com), Nitesh Chawla(nchawla@nd.edu), Neil Shah(nshah@snap.com), Tong Zhao(tzhao@snapchat.com).
 -->
