python ../src_code/train_teacher_gnn.py --datasets=cora --encoder=sage --runs=10 --transductive=production
python ../src_code/train_teacher_gnn.py --datasets=citeseer --encoder=sage --runs=10 --transductive=production
python ../src_code/train_teacher_gnn.py --datasets=pubmed --encoder=sage --runs=10 --transductive=production
python ../src_code/train_teacher_gnn.py --datasets=coauthor-cs --encoder=sage --runs=10 --transductive=production
python ../src_code/train_teacher_gnn.py --datasets=coauthor-physics --encoder=sage --runs=10 --transductive=production
python ../src_code/train_teacher_gnn.py --datasets=amazon-computers --encoder=sage --lr=0.001 --runs=10 --transductive=production
python ../src_code/train_teacher_gnn.py --datasets=amazon-photos --encoder=sage --lr=0.001 --runs=10 --transductive=production