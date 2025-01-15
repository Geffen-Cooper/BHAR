#!/bin/bash
cd ..

seeds=(0)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --multisensor_checkpoint_prefix multisensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_multisensor \
            --policy opportunistic \
            --model_type asynchronous_multisensor \
            --architecture "$architecture" \
            --dataset dsads \
            --seed "$seed" \
            --dataset_top_dir ~/Projects/data/dsads \
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
            --policy_batch_size 16 \
            --policy_lr 1e-6 50 \
            --policy_epochs 5 \
            --policy_val_every_epochs 1 \
            --policy_param_init_vals 0. 0. \
            --policy_param_min_vals 0. 0. \
            --policy_param_max_vals 1.5e-4 10000 \
            --finetune_batch_size 32 \
            --finetune_lr 1e-4 \
            --finetune_epochs 5
    done
done

