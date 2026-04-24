#!/bin/bash

# 사용법: ./script.sh
# 여러 데이터셋에 대해 순차적으로 실험을 실행합니다

# 실행할 데이터셋 목록
# datasets=("Sports_and_Outdoors_5" "Video_Games_5" "Douban_Movie")
datasets=("Beauty_5")

echo "다음 데이터셋들에 대해 순차적으로 실험을 실행합니다:"
echo "${datasets[@]}"
echo "=================================================="

# 각 데이터셋에 대해 반복
for dataset in "${datasets[@]}"
do
    echo "현재 실행 중: dataset=$dataset"
    # echo "시작 시간: $(date)"
    
    python jodie.py --dataset=$dataset --gpu=1 --seed=0
    
    echo "dataset=$dataset 실행 완료"
    # echo "완료 시간: $(date)"
    echo "=================================================="
done

echo "모든 데이터셋 실험이 완료되었습니다!"
# echo "전체 완료 시간: $(date)"