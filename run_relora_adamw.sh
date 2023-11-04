k=3


# python scripts/data_preprocess/split.py \
#   --dataset_path data/gpt4_v2/gpt4_v2_text_only.json \
#   --output_path  data/gpt4_v2_split \
#   --seed 1 \
#   --k ${k}


# run_name="con_lora_llama2-7b_adamw_gpt4_v2_4e-4_ft_weight_norm_gradient_norm"
# lr=4e-4
# bs=30
# per_device_eval_batch_size=1
# use_lora=1
# epochs=1
# gradient_checkpointing=true
# gradient_accumulation_steps=1
# lora_r=8
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# # model_name_or_path=gpt2
# model_name_or_path=meta-llama/Llama-2-7b-hf
# # model_name_or_path=microsoft/phi-1_5
# exp_name="${run_name}_0";
# # data_path="data/c4_10G_split/split_0/"
# data_path="data/gpt4_v2_split/split_0"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# for i in $(seq 1 $((k-1))) 
# do
#     echo "current ${i} split"
#     model_name_or_path=./output_models/${exp_name}
#     selected_portion=$((i+1))
#     echo ${selected_portion}
#     exp_name="${run_name}_${i}"
#     data_path="data/gpt4_v2_split/split_${i}/"
#     bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"
# done

# # worked phi 1.5 continue lora 
# run_name="con_lora_phi1_5_adamw_gpt4_v2_6e-5_ft_per_layer"
# lr=6e-5
# bs=15
# per_device_eval_batch_size=1
# use_lora=1
# epochs=1
# gradient_checkpointing=False
# gradient_accumulation_steps=1
# lora_r=8
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# # model_name_or_path=gpt2
# # model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=microsoft/phi-1_5
# exp_name="${run_name}_0";
# # data_path="data/c4_10G_split/split_0/"
# data_path="data/gpt4_v2_split/split_0"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# for i in $(seq 1 $((k-1))) 
# do
#     echo "current ${i} split"
#     model_name_or_path=./output_models/${exp_name}
#     selected_portion=$((i+1))
#     echo ${selected_portion}
#     exp_name="${run_name}_${i}"
#     data_path="data/gpt4_v2_split/split_${i}/"
#     bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"
# done


# worked gptj_6b continue lora 
# run_name="con_lora_gptj_6b_adamw_gpt4_v2_6e-5_ft_per_layer"
# lr=6e-5
# bs=4
# per_device_eval_batch_size=1
# use_lora=1
# epochs=1
# gradient_checkpointing=False
# gradient_accumulation_steps=1
# lora_r=8
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# # model_name_or_path=gpt2
# # model_name_or_path=meta-llama/Llama-2-7b-hf
# # model_name_or_path=microsoft/phi-1_5
# model_name_or_path=EleutherAI/gpt-j-6b
# exp_name="${run_name}_0";
# # data_path="data/c4_10G_split/split_0/"
# data_path="data/gpt4_v2_split/split_0"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# for i in $(seq 1 $((k-1))) 
# do
#     echo "current ${i} split"
#     model_name_or_path=./output_models/${exp_name}
#     selected_portion=$((i+1))
#     echo ${selected_portion}
#     exp_name="${run_name}_${i}"
#     data_path="data/gpt4_v2_split/split_${i}/"
#     bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"
# done

# gpt2 
k=3
run_name="con_lora_gpt2_adamw_gpt4_v2_2e-3_ft_print_detail_layer"
lr=2e-3
bs=30
per_device_eval_batch_size=1
use_lora=1
epochs=1
gradient_checkpointing=true
gradient_accumulation_steps=1
lora_r=8
block_size=512
ds_config=configs/ds_config_zero2_custom_optimizer.json
model_name_or_path=gpt2
# model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=microsoft/phi-1_5
exp_name="${run_name}_0";
# data_path="data/c4_10G_split/split_0/"
data_path="data/gpt4_v2_split/split_0"
# data_path="data/gpt4_v2"
warmup_ratio=0.01
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:1,2,3,4,5,6,7"  

for i in $(seq 1 $((k-1))) 
do
    echo "current ${i} split"
    model_name_or_path=./output_models/${exp_name}
    selected_portion=$((i+1))
    echo ${selected_portion}
    exp_name="${run_name}_${i}"
    data_path="data/gpt4_v2_split/split_${i}/"
    bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:1,2,3,4,5,6,7"
done

# GPT2 XL 
# run_name="con_lora_gpt2_xl_adamw_gpt4_v2_1e-3_ft_per_layer_print_detail_layer"
# lr=1e-3
# bs=30
# per_device_eval_batch_size=1
# use_lora=1
# epochs=1
# gradient_checkpointing=true
# gradient_accumulation_steps=1
# lora_r=8
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# model_name_or_path=gpt2-xl
# # model_name_or_path=meta-llama/Llama-2-7b-hf
# # model_name_or_path=microsoft/phi-1_5
# exp_name="${run_name}_0";
# # data_path="data/c4_10G_split/split_0/"
# data_path="data/gpt4_v2_split/split_0"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  
# for i in $(seq 1 $((k-1))) 
# do
#     echo "current ${i} split"
#     model_name_or_path=./output_models/${exp_name}
#     selected_portion=$((i+1))
#     echo ${selected_portion}
#     exp_name="${run_name}_${i}"
#     data_path="data/gpt4_v2_split/split_${i}/"
#     bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"
# done