#!/bin/bash

# 사용법: ./script.sh
# 여러 데이터셋에 대해 순차적으로 실험을 실행합니다


python DGCF.py --dataset=wikipedia --sample_length=100 --gpu=0 --seed=0
python DGCF.py --dataset=video --gpu=0 --seed=0 --sample_length=100
python DGCF.py --dataset=lastfm --gpu=0 --seed=0 --sample_length=100
python DGCF.py --dataset=douban_movie --gpu=0 --seed=0 --sample_length=100
python DGCF.py --dataset=mooc --gpu=0 --seed=0 --sample_length=50
python DGCF.py --dataset=yoochoosebuy --sample_length=100 --gpu=0 --seed=0
