#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export HF_ENDPOINT=https://hf-mirror.com

output_dir="Qwen2.5-0.5B-Instruct-sft"
if [ ! -d "./output/${output_dir}" ]; then
  mkdir -p "./output/${output_dir}"
fi

cp $0 "./output/${output_dir}"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port=12220 src/open_r1/sft.py \
    --model_name_or_path /data/liangjh/model_set/Qwen2.5-0.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --dataset_num_proc 60 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --output_dir "./output/${output_dir}" \
    --bf16 \
    --report_to "tensorboard" \
    2>&1 | tee "./output/${output_dir}/${output_dir}.log" &