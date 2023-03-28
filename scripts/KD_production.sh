python ../src_code/main_production.py --datasets=cora --KD_f=0 --KD_kl=0.001 --KD_p=0 --KD_r=0.01 --True_label=1000 --dropout=0.5 --hops=3 --lr=0.01 --margin=0.05 --ns_rate=3 --rw_step=3 --runs=10
python ../src_code/main_production.py --datasets=citeseer --KD_f=0 --KD_kl=0 --KD_p=0 --KD_r=0.0001 --True_label=100 --dropout=0.5 --hops=1 --lr=0.01 --margin=0.1 --ns_rate=5 --rw_step=3 --runs=10
python ../src_code/main_production.py --datasets=pubmed --KD_f=0 --KD_kl=0.01 --KD_p=0 --KD_r=0.001 --True_label=0.0001 --dropout=0.0 --hops=3 --lr=0.01 --margin=0.2 --ns_rate=5 --rw_step=3 --runs=10
python ../src_code/main_production.py --datasets=coauthor-cs --KD_f=0 --KD_kl=10 --KD_p=0 --KD_r=100 --True_label=1 --dropout=0.0 --hops=1 --lr=0.001 --margin=0.1 --ns_rate=4 --rw_step=3 --runs=10
python ../src_code/main_production.py --datasets=coauthor-physics --KD_f=0 --KD_kl=10 --KD_p=0 --KD_r=0.01 --True_label=0.1 --dropout=0.0 --hops=2 --lr=0.0005 --margin=0.2 --ns_rate=4 --rw_step=2 --runs=10
python ../src_code/main_production.py --datasets=amazon-computers --KD_f=0 --KD_kl=0 --KD_p=0 --KD_r=0.1 --True_label=0.01 --dropout=0.0 --hops=2 --lr=0.001 --margin=0.2 --ns_rate=4 --rw_step=2 --runs=10
python ../src_code/main_production.py --datasets=amazon-photos --KD_f=0 --KD_kl=0.0001 --KD_p=0 --KD_r=1000 --True_label=100 --dropout=0.0 --hops=2 --lr=0.001 --margin=0.05 --ns_rate=2 --rw_step=3 --runs=10