
# first use vllm client
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 trl vllm-serve --model /data/liangjh/model_set/Qwen2.5-0.5B-Instruct

# train_grpo.sh
