#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_classifier.py \
            --logging_prefix multisensor_classifier_window_25_acc \
            --architecture "$architecture" \
            --dataset opportunity \
            --seed "$seed" \
            --subjects 1 2 3 4 \
            --sensors acc \
            --body_parts BACK RUA RLA LUA LLA L-SHOE R-SHOE \
            --activities 0 1 2 3 4 \
            --val_frac 0.1 \
            --window_size 25 \
            --overlap_frac 0.5 \
            --batch_size 256 \
            --lr 0.0001 \
            --epochs 25 \
            --ese 10 \
            --log_freq 200
    done
done

