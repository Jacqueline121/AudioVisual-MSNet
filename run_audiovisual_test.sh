#!/usr/bin/env bash

# Test A/V-MSNet audiovisual models

base_path='./experiments/audiovisual_test'

results_path='results'
mkdir -p ${base_path}'/'${results_path}

python3 main.py --gpu_devices 0,1,2,3 --batch_size 128 --n_threads 12 \
    --audiovisual --checkpoint 20 --n_epochs 60 \
    --no_train --no_val \
    --resume_path ./data/pretrained_models/av-msnet_audiovisual/audiovisual_save_60.pth \
    --root_path ${base_path} --result_path ${results_path} \
	--annotation_path_movie50_train ./data/fold_lists/Movie50_list_train_fps.txt \
	--annotation_path_movie50_test ./data/fold_lists/Movie50_list_test_fps.txt \
	--annotation_path_tvsum_train ./data/fold_lists/TVSum_list_train_fps.txt \
	--annotation_path_tvsum_test ./data/fold_lists/TVSum_list_test_fps.txt
