for dataset in cora citeseer pubmed coauthor-cs coauthor-physics
do
    for encoder in sage
    do
        python ../src_code/train_teacher_gnn.py --datasets=${dataset} --encoder=${encoder} --runs=10 --lr=0.005 --transductive=transductive
    done
done

for dataset in amazon-photos amazon-computers
do
    for encoder in sage
    do
        python ../src_code/train_teacher_gnn.py --datasets=${dataset} --encoder=${encoder} --runs=10 --lr=0.001 --transductive=transductive
    done
done

python ../src_code/train_teacher_gnn.py --datasets=coauthor-physics --encoder=sage --runs=10 --lr=0.005 --transductive=transductive