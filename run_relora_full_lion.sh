# k=1
# run_name="phi1_5_lion_gpt4_v2_6e-6_detail_ft_per_layer"
# lr=6e-6
# bs=15
# per_device_eval_batch_size=1
# use_lora=0
# epochs=1
# gradient_checkpointing=False
# gradient_accumulation_steps=1
# lora_r=128
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# # model_name_or_path=configs/llama_350m.json
# # model_name_or_path=pinkmanlove/llama-7b-hf
# # model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=microsoft/phi-1_5

# # model_name_or_path=gpt2
# exp_name="${run_name}_0";
# data_path="data/gpt4_v2"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Lion
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 




# k=1
# run_name="llama2-7b_lion_gpt4_v2_4e-6_detail_ft_per_datail_layer"
# lr=4e-6
# bs=30
# per_device_eval_batch_size=1
# use_lora=0
# epochs=1
# gradient_checkpointing=True
# gradient_accumulation_steps=1
# lora_r=128
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# # model_name_or_path=configs/llama_350m.json
# # model_name_or_path=pinkmanlove/llama-7b-hf
# model_name_or_path=meta-llama/Llama-2-7b-hf
# # model_name_or_path=microsoft/phi-1_5

# # model_name_or_path=gpt2
# exp_name="${run_name}_0";
# data_path="data/gpt4_v2"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Lion
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 

# k=1
# run_name="gpt2_lion_gpt4_v2_1e-5_ft_print_detail_layer"
# lr=1e-5
# bs=30
# per_device_eval_batch_size=1
# use_lora=0
# epochs=1
# gradient_checkpointing=True
# gradient_accumulation_steps=1
# lora_r=128
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# # model_name_or_path=configs/llama_350m.json
# # model_name_or_path=pinkmanlove/llama-7b-hf
# # model_name_or_path=meta-llama/Llama-2-7b-hf
# # model_name_or_path=microsoft/phi-1_5
# model_name_or_path=gpt2
# exp_name="${run_name}_0";
# data_path="data/gpt4_v2"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Lion
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 


k=1
run_name="gpt2_xl_lion_gpt4_v2_1e-5_ft_per_layer_print_detail_layer"
lr=1e-5
bs=30
per_device_eval_batch_size=1
use_lora=0
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=128
block_size=512
ds_config=configs/ds_config_zero2_custom_optimizer.json
# model_name_or_path=configs/llama_350m.json
# model_name_or_path=pinkmanlove/llama-7b-hf
# model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=microsoft/phi-1_5
model_name_or_path=gpt2-xl
exp_name="${run_name}_0";
data_path="data/gpt4_v2"
warmup_ratio=0.01
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Lion
bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 
