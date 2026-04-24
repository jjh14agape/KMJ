#!/bin/bash

# 사용법: ./script.sh
# 여러 데이터셋에 대해 순차적으로 실험을 실행합니다


python main.py --dataset=wikipedia --gpu=0 --num_layer_kf=5 --seed=0
python main.py --dataset=video --gpu=0 --num_layer_kf=4 --seed=0
python main.py --dataset=lastfm --num_layer_kf=1 --gpu=0 --seed=0
python main.py --dataset=douban_movie --gpu=0 --num_layer_kf=1 --seed=0
python main.py --dataset=mooc --gpu=0 --num_layer_kf=1 --seed=0
python main.py --dataset=yoochoosebuy --num_layer_kf=2 --gpu=0 --seed=0