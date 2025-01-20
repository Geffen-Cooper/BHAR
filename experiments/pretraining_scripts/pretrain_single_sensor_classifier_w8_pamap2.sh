#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")
bodyparts=("hand" "chest" "ankle")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      for bp in "${bodyparts[@]}"; do
        python train_har_classifier.py \
              --logging_prefix "single_sensor_classifier_window_8_acc_${bp}" \
              --architecture "$architecture" \
              --dataset pamap2 \
              --seed "$seed" \
              --subjects 1 2 3 4 5 6 7 8 9 \
              --sensors acc \
              --body_parts "$bp" \
              --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
              --val_frac 0.1 \
              --window_size 8 \
              --overlap_frac 0.5 \
              --batch_size 256 \
              --lr 0.0001 \
              --epochs 25 \
              --ese 10 \
              --log_freq 200
            
      done
    done
done

