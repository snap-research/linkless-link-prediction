python ../src_code/main.py --datasets=cora --KD_RM=0 --LLP_D=0.001 --KD_LM=0 --LLP_R=0.01 --True_label=1000 --dropout=0.5  --encoder=sage --hops=3 --lr=0.01 --margin=0.05 --ns_rate=3 --rw_step=3 --runs=10 --transductive=production 
python ../src_code/main.py --datasets=citeseer --KD_RM=0 --LLP_D=0 --KD_LM=0 --LLP_R=0.0001 --True_label=100 --dropout=0.5  --encoder=sage --hops=1 --lr=0.01 --margin=0.1 --ns_rate=5 --rw_step=3 --runs=10 --transductive=production
python ../src_code/main.py --datasets=pubmed --KD_RM=0 --LLP_D=0.01 --KD_LM=0 --LLP_R=0.001 --True_label=0.0001 --dropout=0.0  --encoder=sage --hops=3 --lr=0.01 --margin=0.2 --ns_rate=5 --rw_step=3 --runs=10 --transductive=production
python ../src_code/main.py --datasets=coauthor-cs --KD_RM=0 --LLP_D=10 --KD_LM=0 --LLP_R=100 --True_label=1 --dropout=0.0  --encoder=sage --hops=1 --lr=0.001 --margin=0.1 --ns_rate=4 --rw_step=3 --runs=10 --transductive=production
python ../src_code/main.py --datasets=coauthor-physics --KD_RM=0 --LLP_D=10 --KD_LM=0 --LLP_R=0.01 --True_label=0.1 --dropout=0.0  --encoder=sage --hops=2 --lr=0.0005 --margin=0.2 --ns_rate=4 --rw_step=2 --runs=10 --transductive=production
python ../src_code/main.py --datasets=amazon-computers --KD_RM=0 --LLP_D=0 --KD_LM=0 --LLP_R=0.1 --True_label=0.01 --dropout=0.0  --encoder=sage --hops=2 --lr=0.001 --margin=0.2 --ns_rate=4 --rw_step=2 --runs=10 --transductive=production
python ../src_code/main.py --datasets=amazon-photos --KD_RM=0 --LLP_D=0.0001 --KD_LM=0 --LLP_R=1000 --True_label=100 --dropout=0.0  --encoder=sage --hops=2 --lr=0.001 --margin=0.05 --ns_rate=2 --rw_step=3 --runs=10 --transductive=production
