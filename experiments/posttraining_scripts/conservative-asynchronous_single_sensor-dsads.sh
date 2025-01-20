#!/bin/bash
cd ..

seeds=(2)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix conservative-asynchronous_single_sensor_batch_2 \
            --policy conservative \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset dsads \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts torso right_arm left_arm right_leg left_leg \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
            --val_frac 0.1 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --policy_batch_size 2 \
            --policy_lr 20 20 \
            --policy_epochs 50 \
            --policy_val_every_epochs 10 \
            --policy_param_init_vals 0. 0. \
            --policy_param_min_vals -1000 -1000 \
            --policy_param_max_vals 1000 1000
    done
done

