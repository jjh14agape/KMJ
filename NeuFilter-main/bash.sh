#!/bin/bash

# 사용법: ./script.sh wikipedia
# 첫 번째 인자로 dataset 이름을 받습니다

datasets=("Beauty_5")

echo "sample_length 실험 시작 (dataset=$dataset)"
echo "======================================="

echo "Dataset: $dataset"
echo "num_layer_kf를 1부터 5까지 순차적으로 실행합니다..."
# echo "=================================================="

# echo "현재 실행 중: num_layer_kf=1"
    
# python main.py --dataset=Douban_Movie --gpu=0 --num_layer_kf=1 --seed=0

# echo "num_layer_kf=1 실행 완료"
# echo "=================================================="

# num_layer_kf를 1부터 5까지 반복
num_layer=3
while [ $num_layer -le 5 ]
do
    echo "현재 실행 중: num_layer_kf=$num_layer"
    
    # python main.py --dataset=Beauty_5 --gpu=1 --num_layer_kf=$num_layer --seed=0
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python main.py --dataset=Beauty_5 --gpu=0 --num_layer_kf=$num_layer --seed=0
    
    echo "num_layer_kf=$num_layer 실행 완료"
    echo "=================================================="
    
    (( num_layer+=1 ))
done

echo "모든 실험 완료"



# dataset=$1

# # dataset이 입력되지 않았을 경우 에러 처리
# if [ -z "$dataset" ]; then
#     echo "사용법: $0 <dataset_name>"
#     echo "예시: $0 wikipedia"
#     exit 1
# fi

# echo "Dataset: $dataset"
# echo "num_layer_kf를 1부터 5까지 순차적으로 실행합니다..."
# echo "=================================================="

# # num_layer_kf를 1부터 5까지 반복
# num_layer=1
# while [ $num_layer -le 5 ]
# do
#     echo "현재 실행 중: num_layer_kf=$num_layer"
    
#     python main.py --dataset=$dataset --gpu=0 --num_layer_kf=$num_layer --seed=0
    
#     echo "num_layer_kf=$num_layer 실행 완료"
#     echo "=================================================="
    
#     (( num_layer+=1 ))
# done

# echo "모든 실험이 완료되었습니다!"