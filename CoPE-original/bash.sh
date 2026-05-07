#!/bin/bash

# 사용법: bash run_cope_exp.sh

datasets=("yelp" "Instant_Video")
seeds=(0)

for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "현재 실행 중: dataset=$dataset, seed=$seed"
        
        # python run_cope.py --dataset "$dataset" --cuda 0
        python run_cope.py --dataset "$dataset" --cuda 0 --fast_eval --seed 0

        echo "dataset=$dataset, seed=$seed 실행 완료"
        echo "=================================================="
    done
done

echo "모든 CoPE 실험이 완료되었습니다!"