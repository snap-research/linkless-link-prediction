# Linkless Link Prediction via Relational Distillation

## Introduction
This repository contains the source code for the paper: [Linkless Link Prediction via Relational Distillation](https://arxiv.org/pdf/2210.05801.pdf) 
<img width="1209" alt="image" src="https://user-images.githubusercontent.com/69767476/193711518-fdc8c163-7bbc-4118-ad55-75835954d2c7.png">

## Requirements
Please run the following code to install all the requirements:
```
pip install -r requirements.txt
```

## Usage
### Transductive Setting 
- **Teacher GNN training.** You can change "sage" to "mlp" to obtain supervised training results with MLP. 
```
python train_teacher_gnn.py --datasets=cora --encoder=sage --transductive=transductive
```
To reproduce the supervised results shown in Table 2, you can just simply run the following command. The results will be shown in results/.
```
cd scripts/
bash supervised_transductive.sh
```
- **Student MLP training.** LLP_D and LLP_R indicate the weights for the distribution-based and rank-based matching KD, respectively.
```
python main.py --datasets=cora --LLP_D=1 --LLP_R=1 --True_label=1 --transductive=transductive
```
To reproduce the results shown in Table 2, please run the following command:
```
cd scripts/
bash KD_transductive.sh
```
### Production Setting
- **Pre-process dataset**
In this work, we design a new production setting to resemble the real-world link prediction scenario. This setting mimics practical link prediction use cases. Under the production setting, the newly occurred nodes and edges that can not be seen during the training stage would appear in the graph at inference time. For more details, please refer to our paper Appendix C.2. If you want to apply this setting on our own datasets or split the datsets by your self, please change the dataset name ("dataset") in Line 194 in generate_production_split.py file and run the following command:
```
python generate_production_split.py
```
- **Teacher GNN training.** Note: changing "sage" to "mlp" can reproduce the supervised training results with MLP.
```
python train_teacher_gnn.py --datasets=cora --encoder=sage --transductive=production
```
To reproduce the supervised results shown in Table 3, you can just simply run the following command. The results will be shown in results/.
```
cd scripts/
bash supervised_production.sh
```
- **Student MLP training.** LLP_D and LLP_R indicate the weights for the distribution-based and rank-based matching KD, respectively.
```
python main.py --datasets=cora --LLP_D=1 --LLP_R=1 --True_label=1 --transductive=production
```
To reproduce the results shown in Table 3, please run the following command:
```
cd scripts/
bash KD_production.sh
```
### Reproducing Paper Results
In our experiments, we found that the link prediction performance (when evaluated with Hits@K) of models can greatly vary even when run with the same hyperparameters. Besides, the performance of our method is sensitive to the teacher GNN. Therefore, as mentioned in our paper, we run a hyperparameter sweep for each setting and report the results from the best-performing model (as measured by validation Hits@K).

We conducted a random search across the hyperparameters with [Weights & Biases](https://wandb.ai/home). The sweep configuration files can be found [here](./configurations/).

## Reference
If you find our work useful, please cite the following:
```bibtex
@inproceedings{guo2023linkless,
  title={Linkless link prediction via relational distillation},
  author={Guo, Zhichun and Shiao, William and Zhang, Shichang and Liu, Yozen and Chawla, Nitesh V and Shah, Neil and Zhao, Tong},
  booktitle={International Conference on Machine Learning},
  pages={12012--12033},
  year={2023},
  organization={PMLR}
}
```

## Contact
Please contact zguo5@nd.edu if you have any questions.
