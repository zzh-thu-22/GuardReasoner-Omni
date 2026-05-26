
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MIN_PIXELS=3136 \
MAX_PIXELS=1605632 \
VIDEO_MIN_PIXELS=3136 \
VIDEO_MAX_PIXELS=50176 \
FPS_MIN_FRAMES=1 \
FPS_MAX_FRAMES=128 \
FPS=1.0 \
ENABLE_AUDIO_OUTPUT=false \
USE_AUDIO_IN_VIDEO=false \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
    --model your_model_path \
    --tuner_type full \
    --dataset your_data_path \
    --load_from_cache_file false \
    --add_non_thinking_prefix false \
    --split_dataset_ratio 0.0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 48 \
    --output_dir output_SFT/7B \
    --save_strategy epoch \
    --logging_steps 5 \
    --max_length 24576 \
    --warmup_ratio 0.03 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --report_to swanlab \
    --swanlab_project Omni_SFT \
    --swanlab_exp_name 7B \
    --swanlab_mode cloud \
    --swanlab_token xxx \
    --deepspeed zero2 \
    --attn_impl flash_attention_2 