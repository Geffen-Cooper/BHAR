#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_classifier.py \
            --logging_prefix multisensor_classifier_window_8_acc \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 9 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.1 \
            --window_size 8 \
            --overlap_frac 0.5 \
            --batch_size 256 \
            --lr 0.0001 \
            --epochs 25 \
            --ese 10 \
            --log_freq 200
            # 0: --- lying
            # 1: --- sitting
            # 2: --- standing
            # 3: --+ walking
            # 4: +++ running
            # 5: -++ cycling
            # 6: --+ nordic walking
            # 7: --+ ascending stairs
            # 8: --+ descending stairs
            # 9: --+ vacuuming
            # 10: --+ ironing
            # 11: +++ rope jumping
    done
done

