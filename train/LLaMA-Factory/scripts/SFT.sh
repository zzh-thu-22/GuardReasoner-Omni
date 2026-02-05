device="0,1,2,3"
save_path="./train/saves_SFT/4B"

model="your_model_path"
batch_size=1
cutoff_len=32768

dir="./train/LLaMA-Factory/data"

export SWANLAB_API_KEY=your_swanlab_api_key


CUDA_VISIBLE_DEVICES=$device llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path $model \
    --preprocessing_num_workers 96 \
    --finetuning_type full \
    --template qwen3_vl_nothink \
    --flash_attn auto \
    --dataset_dir $dir \
    --dataset GuardReasoner-OmniTrain \
    --cutoff_len $cutoff_len \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps 48 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_strategy epoch \
    --warmup_ratio 0.03 \
    --packing False \
    --report_to none \
    --output_dir $save_path \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --video_fps 2.0 \
    --video_maxlen 128 \
    --video_max_pixels 65536 \
    --use_swanlab true \
    --swanlab_project GuardReasoner-Omni-SFT \
    --swanlab_run_name 4B \
    --deepspeed ./train/LLaMA-Factory/examples/deepspeed/ds_z2_config.json