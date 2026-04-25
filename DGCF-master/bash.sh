#!/bin/bash

# 사용법: ./script.sh
# 여러 데이터셋에 대해 순차적으로 실험을 실행합니다

# 실행할 데이터셋 목록
# datasets=("video" "tools" "wikipedia" "mooc" "reddit" "douban_movie" "lastfm")
# datasets=("videogames" "garden") #("video" "tools" "douban_movie" "lastfm")

datasets=("movie")

sample_lengths=(100 150 200)

echo "sample_length 실험 시작 (dataset=$dataset)"
echo "======================================="

for sample_length in "${sample_lengths[@]}"
do
    echo "현재 실행 중: sample_length=$sample_length"

    python DGCF.py --dataset=movie --gpu=1 --seed=0 --sample_length=$sample_length
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python DGCF.py --dataset=movie --gpu=0 --seed=0 --sample_length=$sample_length

    echo "완료: sample_length=$sample_length"
    echo "---------------------------------------"
done

echo "모든 실험 완료"

# echo "다음 데이터셋들에 대해 순차적으로 실험을 실행합니다:"
# echo "${datasets[@]}"
# echo "=================================================="

# # 각 데이터셋에 대해 반복
# for dataset in "${datasets[@]}"
# do
#     echo "현재 실행 중: dataset=$dataset"
#     # echo "시작 시간: $(date)"
    
#     python DGCF.py --dataset=$dataset --gpu=0 --seed=0 --sample_length=100
    
#     echo "dataset=$dataset 실행 완료"
#     # echo "완료 시간: $(date)"
#     echo "=================================================="
# done

# echo "모든 데이터셋 실험이 완료되었습니다!"
# # echo "전체 완료 시간: $(date)"