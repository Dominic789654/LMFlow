# search gpt2 ft adamw freeze 
# k=1
# # run_name="con_lora_gpt2_lionlamb_gpt4_v2_1e-2_print_freeze_last4"
# run_name="ft_gpt2_adamw_gpt4_v2_2e-4_print_freeze_all"
# lr=2e-4
# bs=30
# per_device_eval_batch_size=1
# use_lora=0
# epochs=1
# gradient_checkpointing=True
# gradient_accumulation_steps=1
# lora_r=128
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# model_name_or_path=gpt2
# # model_name_or_path=meta-llama/Llama-2-7b-hf
# # model_name_or_path=microsoft/phi-1_5
# exp_name="${run_name}_0";
# # data_path="data/c4_10G_split/split_0/"
# data_path="data/gpt4_v2/"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# freeze_layers="--freeze_layers 0 1 2 3 4 5 6 7 8 9 10 11"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft_gpt2_adamw_gpt4_v2_2e-4_print_freeze_frist4"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 0 1 2 3"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft_gpt2_adamw_gpt4_v2_2e-4_print_freeze_odd"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 1 3 5 7 9 11"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft_gpt2_adamw_gpt4_v2_2e-4_print_freeze_even"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 0 2 4 6 8 10"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft_gpt2_adamw_gpt4_v2_2e-4_print_freeze_last4"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 8 9 10 11"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft_gpt2_adamw_gpt4_v2_2e-4_print_freeze_middle4"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 4 5 6 7"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft_gpt2_adamw_gpt4_v2_2e-4_print_freeze_random4"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 0 3 6 9"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  


k=1
run_name="ft-llama2-7b_adamw_gpt4_v2_4e-5_per_layer_print"
lr=4e-5
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
model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=microsoft/phi-1_5
# model_name_or_path=gpt2
exp_name="${run_name}_0";
data_path="data/gpt4_v2"
warmup_ratio=0.01
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
freeze_layers="--freeze_layers 0 1 2 3 4 5 6 7 8 9 10 11"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 

bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name}  "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 


run_name="ft-llama2-7b_adamw_gpt4_v2_4e-5_per_layer_freeze_frist4"
exp_name="${run_name}_0";
freeze_layers="--freeze_layers 0 1 2 3"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

run_name="ft-llama2-7b_adamw_gpt4_v2_4e-5_per_layer_freeze_odd"
exp_name="${run_name}_0";
freeze_layers="--freeze_layers 1 3 5 7 9 11"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft-llama2-7b_adamw_gpt4_v2_4e-5_per_layer_freeze_even"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 0 2 4 6 8 10"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft-llama2-7b_adamw_gpt4_v2_4e-5_per_layer_freeze_last4"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 8 9 10 11"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft-llama2-7b_adamw_gpt4_v2_4e-5_per_layer_freeze_middle4"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 4 5 6 7"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  

# run_name="ft-llama2-7b_adamw_gpt4_v2_4e-5_per_layer_freeze_random4"
# exp_name="${run_name}_0";
# freeze_layers="--freeze_layers 0 3 6 9"
# bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${freeze_layers}" "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  


# k=1
# run_name="phi1_5_adamw_gpt4_v2_6e-5_ft_per_layer"
# lr=6e-5
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
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 

# k=1
# run_name="gptj_6b_adamw_gpt4_v2_6e-5_ft_per_layer"
# lr=6e-5
# bs=4
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
# # model_name_or_path=microsoft/phi-1_5
# model_name_or_path=EleutherAI/gpt-j-6b
# # model_name_or_path=gpt2
# exp_name="${run_name}_0";
# data_path="data/gpt4_v2"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 



# k=1
# run_name="llama2-7b_adamw_gpt4_v2_8e-4_per_layer"
# lr=8e-4
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
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 

# k=1
# run_name="gpt2_adamw_gpt4_v2_1e-4_ft_print_detail"
# lr=1e-4
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
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 


# k=1
# run_name="gpt2_xl_adamw_gpt4_v2_1e-4_ft_per_layer_print_detail_layer"
# lr=1e-4
# bs=30
# per_device_eval_batch_size=1
# use_lora=0
# epochs=1
# gradient_checkpointing=True
# gradient_accumulation_steps=1
# lora_r=8
# block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
# # model_name_or_path=configs/llama_350m.json
# # model_name_or_path=pinkmanlove/llama-7b-hf
# # model_name_or_path=meta-llama/Llama-2-7b-hf
# # model_name_or_path=microsoft/phi-1_5
# model_name_or_path=gpt2-xl
# exp_name="${run_name}_0";
# data_path="data/gpt4_v2"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 
