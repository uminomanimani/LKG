teacher_model = "gpt-4o-mini"
dataset = "docred"
max_hop = 2
lr = 5e-5
max_grad_norm = 1.0
epochs = 50
accumulation_steps = 4
warmup_ratio = 0.18
maintein_ratio = 0.09
decay_factor = 0.6
min_lr = 5e-7

python train.py \
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