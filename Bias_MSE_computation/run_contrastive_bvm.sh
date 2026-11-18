#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
run_id=$(date +%Y%m%d-%H%M%S-%3N)
seeds=(41 42 43 44 45)

# Runs are directly under the repository root so ../data resolves correctly.
common_args=(
    --dataset ogbn-arxiv
    --norm_feat false
    --fast true
    --depths 2
    --batch_norm true
    --hidden_dim 256
    --init_batch 1024
    --sample_size 10240
    --scale_factor 5
    --wd 1e-4
    --drop 0.5
    --lr 0.001
    --epochs 1000
    --early_stop 300
    --label_rate 0.025
    --labeled_ratio 0.9
    --alpha_mode optimal
    --sampling_replacement false
    --compute_bvm true
    --bvm_compare_alpha false
    --bvm_replacement true
    --bvm_samples 100
    --bvm_nodes 500
    --bvm_seed 2026
    --bvm_split validation
)

run_condition() {
    local seed=$1 condition=$2 weight=$3
    local run_dir="$repo_root/bvm-ogbn-arxiv-$condition-seed-$seed-$run_id"
    mkdir -- "$run_dir"
    cp -- "$script_dir"/*.py "$run_dir/"
    printf 'Running %s with seed %s in %s\n' "$condition" "$seed" "$run_dir"
    (
        cd -- "$run_dir"
        conda run --no-capture-output -n graphany python ./BVSampler_main.py "${common_args[@]}" \
            --seed "$seed" --con_weight "$weight" --bvm_output bvm.csv
    )
    printf 'Saved %s/bvm.csv\n' "$run_dir"
}

for seed in "${seeds[@]}"; do
    run_condition "$seed" no-contrastive 0
    run_condition "$seed" with-contrastive 0.5
done
