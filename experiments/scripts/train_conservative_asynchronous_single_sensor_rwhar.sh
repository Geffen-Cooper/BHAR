#!/bin/bash
cd ..

seeds=(0)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --checkpoint_prefix classifier_window_8_acc \
            --logging_prefix conservative_asynchronous_single_sensor \
            --policy conservative \
            --architecture "$architecture" \
            --dataset rwhar \
            --seed "$seed" \
            --dataset_top_dir ../../../../store/nt9637/BHAR/data/rwhar \
            --subjects 1 4 5 7 9 10 11 12 13 14 15 \
            --sensors acc \
            --body_parts chest forearm head shin thigh upperarm waist \
            --activities 0 1 2 3 4 5 6 7 \
            --val_frac 0.1 \
            --window_size 8 \
            --overlap_frac 0.5 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --model_type asynchronous_single_sensor \
            --batch_size 16 \
            --lr 1e-6 50 \
            --epochs 5 \
            --val_every_epochs 1 \
            --param_init_vals 0. 0. \
            --param_min_vals 0. 0. \
            --param_max_vals 1.5e-4 10000
    done
done

