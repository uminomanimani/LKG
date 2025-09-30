export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /mnt/HDD/mlk_workspace/chatglm3-6b \
    --adapter_name_or_path ./saves/gpt-4o-mini/docred/student_model_checkpoints/chatglm3-6b  \
    --eval_dataset dev \
    --dataset_dir ./saves/gpt-4o-mini/docred/fine_tuning_for_description_extraction \
    --template chatglm3 \
    --finetuning_type lora \
    --output_dir ./saves/gpt-4o-mini/docred/student_output/chatglm3-6b/dev \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 10240 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 9999 \
    --max_new_tokens 8192 \
    --predict_with_generate \
    --trust_remote_code True \

