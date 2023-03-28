for dataset in cora citeseer pubmed coauthor-cs coauthor-physics
do
    for encoder in sage
    do
        python ../src_code/main_sp.py --datasets=${dataset} --encoder=${encoder} --runs=10 --lr=0.005
    done
done

for dataset in amazon-photos amazon-computers
do
    for encoder in sage
    do
        python ../src_code/main_sp.py --datasets=${dataset} --encoder=${encoder} --runs=10 --lr=0.001
    done
done
