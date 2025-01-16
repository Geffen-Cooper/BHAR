#!/bin/bash
cd ..

seeds=(0)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --checkpoint_prefix classifier_window_25_acc \
            --logging_prefix unconstrained_synchronous_multisensor \
            --policy unconstrained_12 \
            --architecture "$architecture" \
            --dataset rwhar \
            --seed "$seed" \
            --dataset_top_dir ../../../../store/nt9637/BHAR/data/dsads/data \
            --subjects 1 4 5 7 9 10 11 12 13 14 15 \
            --sensors acc \
            --body_parts chest forearm head shin thigh upperarm waist \
            --activities 0 1 2 3 4 5 6 7 \
            --val_frac 0.1 \
            --window_size 25 \
            --overlap_frac 0.5 \
            --harvesting_sensor_window_size 25 \
            --leakage 6.6e-6 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --model_type synchronous_multisensor
    done
done

