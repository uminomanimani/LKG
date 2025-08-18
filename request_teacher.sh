#!/bin/bash
set -e

teacher_model="gpt-4o-mini"
dataset="docred"
n_threads=16
dataset_type=("train" "dev" "test")

for _dataset_type in "${dataset_type[@]}"; do
    # request entity descriptions via teacher model, to which the entities are visible
    python request_entity_description.py \
        --teacher_model "${teacher_model}" \
        --dataset "${dataset}" \
        --n_threads "${n_threads}" \
        --dataset_type "${_dataset_type}"

    # generate alpaca format for supervised fine-tuning
    python ner_sft.py \
        --teacher_model "${teacher_model}" \
        --dataset "${dataset}" \
        --dataset_type "${_dataset_type}"
    
    python build_graph_for_teacher.py \
        --teacher_model "${teacher_model}" \
        --dataset "${dataset}" \
        --dataset_type "${_dataset_type}"
done
