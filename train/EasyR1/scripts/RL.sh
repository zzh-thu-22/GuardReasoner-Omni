#!/usr/bin/env bash
set -x

export DECORD_EOF_RETRY_MAX=2048001
export SWANLAB_API_KEY=your_swanlab_api_key

SCRIPT_DIR=$(cd $(dirname $0); pwd)
EASYR1_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH=$PYTHONPATH:$EASYR1_ROOT

project_name='GuardReasoner-Omni-RL'
exp_name='4B'

MODEL_PATH="your_model_path"
TRAIN_FILE="harm_sample.json"
TEST_FILE="harm_sample.json"
IMAGE_DIR=""

ROLLOUT_BS=256
GLOBAL_BS=128
MB_PER_UPDATE=1
MB_PER_EXP=2
TP_SIZE=4
N_GPUS_PER_NODE=4
NNODES=1


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=./train/EasyR1/examples/config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_dir="${IMAGE_DIR}" \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    data.max_prompt_length=22528 \
    data.max_response_length=2048 \
    data.max_pixels=65536 \
    data.min_pixels=4096 \
    algorithm.disable_kl=true \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=2e-6 \
    worker.rollout.n=16 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs=100 \
    trainer.save_freq=4 \
    trainer.save_checkpoint_path=./train/saves_RL