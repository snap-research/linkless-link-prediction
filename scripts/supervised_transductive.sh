for dataset in cora citeseer pubmed coauthor-cs coauthor-physics
do
    for encoder in sage
    do
        python ../src/train_teacher_gnn.py --datasets=${dataset} --encoder=${encoder} --runs=10 --lr=0.005 --transductive=transductive
    done
done

for dataset in amazon-photos amazon-computers
do
    for encoder in sage
    do
        python ../src/train_teacher_gnn.py --datasets=${dataset} --encoder=${encoder} --runs=10 --lr=0.001 --transductive=transductive
    done
done
python ../src/train_teacher_gnn.py --datasets=collab --encoder=sage --num_layers=3 --runs=10 --lr=0.005 --transductive=transductive

python ../src/train_teacher_gnn.py --datasets=cora --encoder=sage --runs=10 --lr=0.005 --transductive=transductive