#!/bin/bash

# 사용법: bash run_cope_exp.sh

datasets=("Instant_Video")
seeds=(0)

echo "다음 데이터셋들에 대해 CoPE 실험을 실행합니다:"
echo "${datasets[@]}"
echo "=================================================="

for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "현재 실행 중: dataset=$dataset, seed=$seed"
        
        python run_cope.py \
            --dataset "$dataset" \
            --cuda 0 \
            --seed "$seed" \
            --epochs 50 \
            --fast_eval

        echo "dataset=$dataset, seed=$seed 실행 완료"
        echo "=================================================="
    done
done

echo "모든 CoPE 실험이 완료되었습니다!"