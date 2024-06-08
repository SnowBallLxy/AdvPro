#!bin/bash
model_list=(codegen-6B-mono codegen2-7B)

for model in ${model_list[@]}; do
    echo "Running $model"
    python3 run_test_LLM.py --find_adv --find_mode full --model_name_or_path $model --max_source_length 924 --max_target_length 64 --beam_size 1 --eval_batch_size 1 --file_idx all --gpu_idx 0
done