#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("attend")
bodyparts=("torso" "right_arm" "left_arm" "right_leg" "left_leg")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      for bp in "${bodyparts[@]}"; do
        python train_har_classifier.py \
              --logging_prefix "single_sensor_classifier_window_8_acc_${bp}" \
              --architecture "$architecture" \
              --dataset dsads \
              --seed "$seed" \
              --dataset_top_dir ~/Projects/data/dsads \
              --subjects 1 2 3 4 5 6 7 8 \
              --sensors acc \
              --body_parts "$bp" \
              --activities 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
              --val_frac 0.1 \
              --window_size 8 \
              --overlap_frac 0.5 \
              --batch_size 256 \
              --lr 0.0001 \
              --epochs 25 \
              --ese 10 \
              --log_freq 200
              # 0: --- sitting
              # 1: --- standing
              # 2: --- lying on back
              # 3: --- lying on right side
              # 4: +++ ascending stairs
              # 5: +++ descending stairs
              # 6: --- standing in elevator
              # 7: +--  moving in elevator
              # 8: +++ walking in parking lot
              # 9: +++ walking on flat treadmill
              # 10: +++ walking on inclined treadmill
              # 11: +++ running on treadmill
              # 12: +++ exercising on stepper
              # 13: +++ exercising on cross trainer
              # 14: +++ cycling on exercise bike horizontal
              # 15: +++ cycling on exercise bike vertical
              # 16: +-- rowing
              # 17: +++ jumping
              # 18: +++ playing basketball
            
      done
    done
done

