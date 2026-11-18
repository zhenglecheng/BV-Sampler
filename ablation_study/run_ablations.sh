
cd ./ablation_study_au_1_structure_only

python  -u BVSampler_main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --con_weight 0.5 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000 > arxiv_batch_au_1_structure_only.log 2>&1

cd ../ablation_study_biased_weights_structure_only

python  -u BVSampler_main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --con_weight 0.5 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000 > arxiv_batch_biased_weight_structure_only.log 2>&1

cd ../ablation_study_DGI

python  -u BVSampler_main_DGI.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --con_weight 0.5 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000 > arxiv_batch_DGI.log 2>&1

cd ../ablation_study_au_1

python  -u BVSampler_main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --con_weight 0.5 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000 > arxiv_batch_au_1.log 2>&1

cd ../ablation_study_graphcl

python  -u BVSampler_main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --con_weight 0.5 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000 > arxiv_batch_graphcl.log 2>&1

cd ../ablation_study_NO_infoNCE

python  -u BVSampler_main.py --dataset ogbn-arxiv --fast true --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --con_weight 0 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000 > arxiv_batch_no_infonce.log 2>&1






