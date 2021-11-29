#!/bin/bash
###
 # @Author: jianzhnie
 # @Date: 2021-11-29 16:49:02
 # @LastEditTime: 2021-11-29 17:13:43
 # @LastEditors: jianzhnie
 # @Description:
 # 
### 

python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/hymenoptera/split/train \
    --output_path /home/robin/jianzh/AutoTimm/benchmark \
    --report_path /home/robin/jianzh/AutoTimm/benchmark \
    --dataset  hymenoptera \
    --model_config  'default' \
    --batch-size 32 \
    --num_epochs 1 \
    --num_trials 1 \
    --proxy \
    --train_framework autogluon 

python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/hymenoptera/split/train \
    --output_path /home/robin/jianzh/AutoTimm/benchmark \
    --report_path /home/robin/jianzh/AutoTimm/benchmark \
    --dataset  hymenoptera \
    --model_config  'default' \
    --batch-size  16 \
    --num_epochs 1 \
    --num_trials 1 \
    --proxy \
    --train_framework autotimm

## use docker 
python tools/benchmark.py \
    --data_path /home/image_data/hymenoptera/split/train \
    --output_path /home/robin/jianzh/AutoTimm/benchmark \
    --report_path /home/robin/jianzh/AutoTimm/benchmark \
    --dataset  hymenoptera \
    --model_config  'default_hpo' \
    --batch-size 64 \
    --num_epochs 10 \
    --num_trials 4 \
    --train_framework autotimm


## on aiarts
python benchmark.py \
    --data_path /data/AutoML_compete/Flowers-Recognition/split/train \
    --output_path /data/autodl/benchmark \
    --report_path /data/autodl/benchmark \
    --dataset  Flowers-Recognition \
    --batch-size 32 \
    --num_epochs 10 \
    --model_config  'default' \
    --train_framework autotorch 