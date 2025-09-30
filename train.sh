#!/bin/bash
set -e

teacher_model="gpt-4o-mini"
dataset="docred"
n_threads=16
dataset_types=("train" "dev" "test")
max_hop=2
lr=5e-5
max_grad_norm=1.0
epochs=50
accumulation_steps=4
warmup_ratio=0.18
maintein_ratio=0.09
decay_factor=0.6
min_lr=5e-7
student_model="chatglm3-6b"

for dataset_type in "${dataset_types[@]}"; do
    # request entity descriptions via teacher model, to which the entities are visible
    python request_entity_description_multithread.py \
        --teacher_model "${teacher_model}" \
        --dataset "${dataset}" \
        --n_threads "${n_threads}" \
        --dataset_type "${dataset_type}"

    # generate alpaca format for supervised fine-tuning, where the entities are invisible
    python ner_sft.py \
        --teacher_model "${teacher_model}" \
        --dataset "${dataset}" \
        --dataset_type "${dataset_type}"
    
    python build_graph_for_teacher.py \
        --teacher_model "${teacher_model}" \
        --dataset "${dataset}" \
        --dataset_type "${dataset_type}"
done

python train_graph.py \
    --teacher_model "${teacher_model}" \
    --dataset "${dataset}" \
    --max_hop "${max_hop}" \
    --lr "${lr}" \
    --max_grad_norm "${max_grad_norm}" \
    --epochs "${epochs}" \
    --accumulation_steps "${accumulation_steps}" \
    --warmup_ratio "${warmup_ratio}" \
    --maintein_ratio "${maintein_ratio}" \
    --decay_factor "${decay_factor}" \
    --min_lr "${min_lr}"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /mnt/HDD/mlk_workspace/${student_model} \
    --dataset train \
    --dataset_dir "./saves/${teacher_model}/${dataset}/fine_tuning_for_description_extraction" \
    --template chatglm3 \
    --finetuning_type lora \
    --output_dir ./saves/${teacher_model}/${dataset}/student_model_checkpoints/${student_model} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 8192 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --save_strategy epoch \
    --num_train_epochs 8 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --val_size 0 \
    --plot_loss \
    --max_new_tokens 8192 \
    --load_best_model_at_end False \
    --bf16 \
    --trust_remote_code True
