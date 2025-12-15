#!/bin/bash


# 创建日志文件名（包含时间戳）
LOG_FILE="./exp_tucker_10/train_$(date +%Y%m%d_%H%M%S).log"

ln -sf $(which python) ./debug

TIME_LIMIT_SEC=7020

# 确保输出目录存在
mkdir -p ./exp_tucker_10

echo "=========================================="
echo "开始细粒度注意力模型训练"
echo "日志文件: $LOG_FILE"
echo "=========================================="


export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2

nohup ./debug train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property exp_bandgap \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 514 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --weight_decay 5e-3 \
    --warmup_steps 2000 \
    --alignn_layers 5 \
    --gcn_layers 5 \
    --hidden_features 256 \
    --graph_dropout 0.25 \
    --graph_builder kgcnn \
    --preprocessed_dir preprocessed_data \
    --late_fusion_type tucker \
    --late_fusion_output_dim 64 \
    --use_cross_modal False \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --middle_fusion_dropout 0.35 \
    --middle_fusion_use_gate_norm True \
    --middle_fusion_use_learnable_scale True \
    --middle_fusion_initial_scale 1 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --fine_grained_use_projection False \
    --early_stopping_patience 30 \
    --output_dir ./exp_tucker_10 \
    --num_workers 24 \
    --random_seed 42 \
    > "$LOG_FILE" 2>&1 &

echo "=========================================="
echo "训练已在后台启动，PID: $!"
echo "使用以下命令查看进度:"
echo "  tail -f $LOG_FILE"
