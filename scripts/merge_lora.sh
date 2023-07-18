model_name_or_path="$1"
lora_path="$2"
output_path=${lora_path}_merged
python examples/merge_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --lora_model_path ${lora_path} \
    --output_model_path  ${output_path}