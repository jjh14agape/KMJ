#!/bin/bash

# 사용법: ./script.sh
# 여러 데이터셋에 대해 순차적으로 실험을 실행합니다


# python jodie.py --dataset=wikipedia --gpu=0 --seed=0
# python jodie.py --dataset=video --gpu=0 --seed=0
# python jodie.py --dataset=lastfm --gpu=0 --seed=0
# python jodie.py --dataset=douban_movie --gpu=0 --seed=0
python jodie.py --dataset=mooc --gpu=0 --seed=0
python jodie.py --dataset=yoochoosebuy --gpu=0 --seed=0


