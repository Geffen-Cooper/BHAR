#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --policy_logging_prefix conservative_asynchronous-single_sensor \
            --policy conservative \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset rwhar \
            --seed "$seed" \
            --dataset_top_dir ~/Projects/data/rwhar \
            --subjects 11 4 5 7 9 10 11 12 13 14 15 \
            --sensors acc \
            --body_parts chest forearm head shin thigh upperarm waist \
            --activities 0 1 2 3 4 5 6 7 \
            --val_frac 0.1 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --policy_batch_size 16 \
            --policy_lr 1e-6 50 \
            --policy_epochs 5 \
            --policy_val_every_epochs 1 \
            --policy_param_init_vals 0. 0. \
            --policy_param_min_vals 0. 0. \
            --policy_param_max_vals 1.5e-4 10000
    done
done

