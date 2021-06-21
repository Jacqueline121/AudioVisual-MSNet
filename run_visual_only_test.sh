#!/usr/bin/env bash

# Test A/V-MSNet visual only models

base_path='./experiments/visual_only_test'

results_path='results'
mkdir -p ${base_path}'/'${results_path}

python3 main.py --gpu_devices 0,1,2,3 --batch_size 128 --n_threads 12 \
    --checkpoint 20 --n_epochs 60 \
    --no_train --no_val \
    --root_path ${base_path} --result_path ${results_path} \
    --pretrain_path ./data/pretrained_models/av-msnet_visual_only/visual_save_60.pth  \
	--annotation_path_movie50_train ./data/fold_lists/Movie50_av_list_train_fps.txt \
	--annotation_path_movie50_test ./data/fold_lists/Movie50_av_list_test_fps.txt \
	--annotation_path_tvsum_train ./data/fold_lists/TVSum_list_train_fps.txt \
	--annotation_path_tvsum_test ./data/fold_lists/TVSum_list_test_fps.txt
