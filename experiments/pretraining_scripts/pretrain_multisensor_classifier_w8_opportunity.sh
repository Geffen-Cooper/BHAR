#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_classifier.py \
            --logging_prefix multisensor_classifier_window_8_acc \
            --architecture "$architecture" \
            --dataset opportunity \
            --seed "$seed" \
            --subjects 1 2 3 4 \
            --sensors acc \
            --body_parts BACK RUA RLA LUA LLA L-SHOE R_SHOE \
            --activities 0 1 2 3 4 \
            --val_frac 0.1 \
            --window_size 8 \
            --overlap_frac 0.5 \
            --batch_size 256 \
            --lr 0.0001 \
            --epochs 25 \
            --ese 10 \
            --log_freq 200
            # 0: -++ Null
            # 1: --+ Stand
            # 2: --+ Walk
            # 3: --- Sit
            # 4: --- Lie
    done
done

