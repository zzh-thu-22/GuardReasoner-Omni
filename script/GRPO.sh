

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MIN_PIXELS=3136 \
MAX_PIXELS=1605632 \
VIDEO_MIN_PIXELS=3136 \
VIDEO_MAX_PIXELS=50176 \
FPS_MIN_FRAMES=1 \
FPS_MAX_FRAMES=128 \
FPS=1.0 \
ENABLE_AUDIO_OUTPUT=false \
USE_AUDIO_IN_VIDEO=false \
swift rlhf \
    --rlhf_type grpo \
    --model your_model_path \
    --tuner_type full \
    --split_dataset_ratio 0.01 \
    --external_plugins ./ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs my_grpo_reward \
    --reward_weights 1.0 \
    --vllm_mode colocate \
    --use_vllm true \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_limit_mm_per_prompt '{"image": 1, "video": 1, "audio": 1}' \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 24576 \
    --temperature 1.0 \
    --top_p 0.95 \
    --num_generations 8 \
    --torch_dtype bfloat16 \
    --dataset your_data_path \
    --load_from_cache_file false \
    --max_completion_length 4096 \
    --max_length 20480 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-6 \
    --gradient_accumulation_steps 16 \
    --save_steps 112 \
    --eval_steps 112 \
    --save_total_limit 15 \
    --logging_steps 4 \
    --output_dir output_GRPO/7B \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --log_completions true \
    --report_to swanlab \
    --swanlab_project Omni_GRPO \
    --swanlab_exp_name 7B \
    --swanlab_mode cloud \
    --swanlab_token xxx \
    --beta 0.0 \
    --deepspeed zero3