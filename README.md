# Demo Code for BVSampler
---




## Command 


Testing on the PubMed, Reddit and ArXiv Datasets with full graph evaluation in rich label setting (Table 1)
```
python3 main.py --dataset Cora --norm_feat true --fast true --hidden_dim 16 --init_batch 256 --sample_size 400 --wd 5e-4 --seed 42
python3 main.py --dataset PubMed --norm_feat true --fast true --hidden_dim 16 --init_batch 256 --sample_size 400 --early_stop 10 --wd 5e-4 --seed 42 
python3 main.py --dataset Reddit --fast true --norm_feat false --hidden_dim 128 --init_batch 1024 --sample_size 5120 --early_stop 100 --wd 1e-4 --seed 42 
python3 main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 1000 --lr 0.001
```


Testing on the PubMed, Reddit and ArXiv Datasets with full graph evaluation in sparse label setting (Table 2)
```
python3 main.py --dataset PubMed --fast true --norm_feat true --hidden_dim 16 --init_batch 256 --sample_size 400 --early_stop 10 --wd 5e-4 --seed 42 --label_rate 0.05
python3 main.py --dataset Reddit --fast true --norm_feat false --hidden_dim 128 --init_batch 1024 --sample_size 5120 --early_stop 100 --wd 1e-4 --seed 42 --label_rate 0.05
python3 main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 1000 --lr 0.001 --label_rate 0.05
```

Testing on the ArXiv Dataset with batch-only evaluation in sparse label setting (Table 3)
```
python3 main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 500 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --label_rate 0.05 --samp_inference true
python3 main.py --dataset ppi-large --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 128 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --label_rate 0.025 --samp_inference true
```


Ablation Study on ArXiv Dataset (Table 4)
```
cd ablation_study
./run_ablation.sh

```

Bias Varince MSE computation (Table 10)
```
cd Bias_MSE_computation
./run_contrastive_bvm.sh

```
