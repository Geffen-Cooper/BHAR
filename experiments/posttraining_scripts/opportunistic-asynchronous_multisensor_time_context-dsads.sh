#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --multisensor_checkpoint_prefix multisensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_multisensor_time_context \
            --policy opportunistic \
            --model_type asynchronous_multisensor_time_context \
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
            --finetune_batch_size 32 \
            --finetune_lr 1e-4 \
            --finetune_epochs 10
    done
done

