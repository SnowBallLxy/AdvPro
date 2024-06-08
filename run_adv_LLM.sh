#!bin/bash
model_list=(codegen-2B-mono santacoder codegen2-1B)

for model in ${model_list[@]}; do
    echo "Adv $model"
    python3 run_find_adv_target_LLM.py --find_adv --find_mode full --model_name_or_path santacoder --max_source_length 900 --max_target_length 0 --beam_size 1 --eval_batch_size 1 --file_idx all --gpu_idx 0
done