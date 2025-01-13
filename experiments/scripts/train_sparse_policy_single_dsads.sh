#!/bin/bash
cd ..

seeds=(0)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --checkpoint_prefix classifier_window_8_acc \
            --logging_prefix policy_sparse_train \
            --policy conservative \
            --architecture "$architecture" \
            --dataset dsads \
            --seed "$seed" \
            --dataset_top_dir ~/Projects/data/dsads \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts right_leg \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
            --val_frac 0.1 \
            --window_size 8 \
            --overlap_frac 0.5 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --model_type sparse_asychronous_baseline \
            --batch_size 16 \
            --lr 1e-6 5 \
            --epochs 50 \
            --val_every_epochs 1 \
            --param_init_vals 0. 0. \
            --param_min_vals 0. 0. \
            --param_max_vals 1.5e-4 10000
    done
done

