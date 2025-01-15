#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_classifier.py \
            --logging_prefix multisensor_classifier_window_25_acc \
            --architecture "$architecture" \
            --dataset rwhar \
            --seed "$seed" \
            --dataset_top_dir ~/Projects/data/rwhar \
            --subjects 1 4 5 7 9 10 11 12 13 14 15 \
            --sensors acc \
            --body_parts chest forearm head shin thigh upperarm waist \
            --activities 0 1 2 3 4 5 6 7 \
            --val_frac 0.1 \
            --window_size 25 \
            --overlap_frac 0.5 \
            --batch_size 256 \
            --lr 0.0001 \
            --epochs 25 \
            --ese 10 \
            --log_freq 200
            # 0: +-- climbingup
            # 1: +-- climbingdown
            # 2: +++ jumping
            # 3: --- lying
            # 4: +++ running
            # 5: --- sitting
            # 6: --- standing
            # 7: +-- walking
    done
done

