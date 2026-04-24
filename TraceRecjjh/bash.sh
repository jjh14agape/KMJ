#!/bin/bash

# 실행할 path 조합
num_path_us=(10 20 30)
num_path_is=(10 20 30)

echo "다음 조합들에 대해 순차적으로 실험을 실행합니다:"
echo "num_path_u: ${num_path_us[@]}"
echo "num_path_i: ${num_path_is[@]}"
echo "=================================================="

for num_path_u in "${num_path_us[@]}"
do
    for num_path_i in "${num_path_is[@]}"
    do
        echo "현재 실행 중: num_path_u=$num_path_u, num_path_i=$num_path_i"
        echo "-----------------------------------------------"

        python tracerec.py \
            --dataset=ml1m \
            --num_path_u=$num_path_u \
            --num_path_i=$num_path_i \
            --seed=0 \
            --gpu=1 \
            --project

        echo "완료: num_path_u=$num_path_u, num_path_i=$num_path_i"
        echo "=================================================="
    done
done

echo "모든 실험이 완료되었습니다!"