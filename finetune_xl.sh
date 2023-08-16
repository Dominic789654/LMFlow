

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
bash ./scripts/153_ft.sh "--master_port=11000 --include localhost:0,1,2,3,4,5,6,7" > ./log/ft_pretrained_bloom_093.log 2>&1

# bash ./scripts/jipeng_forgetting_gptj.sh "--master_port=10065 --num_gpus=8"

# lr=8e-5
# bs=2
# per_device_eval_batch_size=1
# use_lora=1
# epochs=1
# gradient_checkpointing=False
# gradient_accumulation_steps=16
# lora_r=128
# block_size=512
# ds_config=configs/ds_config_zero2.json
# model_name_or_path="pinkmanlove/llama-7b-hf"
# exp_name="xl_0129";
# data_path="/home/xiangliu/LMFlow/data/high_SFT_data_0501"
# eval_dataset_path="/home/xiangliu/LMFlow/data/eval_760"
# bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# eval_dataset_name=commonsense_qa_eval
# bash ./scripts/run_benchmark_xl.sh ${exp_name} ${eval_dataset_name} ./output_models/${exp_name} "--master_port=10065 --num_gpus=1" 

# eval_dataset_name=all_nll_eval
# bash ./scripts/run_benchmark_xl.sh ${exp_name} ${eval_dataset_name} ./output_models/${exp_name} "--master_port=10065 --num_gpus=1" 


# k=2
# origin_data_path="/home/xiangliu/LMFlow/data/high_SFT_data_0501/sharegpt_en5w_cn1w+belle_cn_8w+gpt4_5w+water_degpt4eval.json"
# split_data_path="/home/xiangliu/LMFlow/data/high_SFT_data_0501_split"
# python3 ./scripts/data_preprocess/split.py  \
#     --dataset_path ${origin_data_path}  \
#     --output_path  ${split_data_path} \
#     --k ${k} \
#     --seed 42


# count=0
# echo "count: $count"
# lr=1e-4
# bs=8
# per_device_eval_batch_size=1
# use_lora=0
# epochs=1
# gradient_checkpointing=True
# gradient_accumulation_steps=1
# lora_r=128
# block_size=2048
# ds_config=configs/ds_config_zero3.json
# model_name_or_path="pinkmanlove/llama-7b-hf"
# exp_name="xl_0130";
# data_path=${split_data_path}/split_${count}/
# eval_dataset_path="/home/xiangliu/LMFlow/data/eval_760"
# bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# eval_dataset_name=commonsense_qa_eval
# bash ./scripts/run_benchmark_xl.sh ${exp_name} ${eval_dataset_name} ./output_models/${exp_name} "--master_port=10065 --num_gpus=1" 

# eval_dataset_name=all_nll_eval
# bash ./scripts/run_benchmark_xl.sh ${exp_name} ${eval_dataset_name} ./output_models/${exp_name} "--master_port=10065 --num_gpus=1" 


# k=4
# origin_data_path="/home/xiangliu/LMFlow/data/continue_wiki_formated_train/train_wiki_formated.json"
# split_data_path="/home/xiangliu/LMFlow/data/continue_wiki_formated_train_split/"
# python3 ./scripts/data_preprocess/split.py  \
#     --dataset_path ${origin_data_path}  \
#     --output_path  ${split_data_path} \
#     --k ${k} \
#     --seed 42

# count=0
# echo "count: $count"
# lr=8e-5
# bs=2
# per_device_eval_batch_size=1
# use_lora=1
# epochs=1
# gradient_checkpointing=False
# gradient_accumulation_steps=16
# lora_r=2048
# block_size=1024
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=bigscience/bloom-7b1
# exp_name="xl_0125";
# data_path=${split_data_path}/split_${count}/
# eval_dataset_path="/home/xiangliu/LMFlow/data/continue_wiki_formated_eval/"
# bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 
# count=`expr $count + 1`

# while [ $count -lt $k ]
# do 
#     echo "count: $count"
#     model_name_or_path=./output_models/${exp_name}
#     data_path=${split_data_path}/split_${count}/
#     bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 
#     count=`expr $count + 1`
# done


# test_dataset_path="/home/xiangliu/LMFlow/data/continue_wiki_formated_test"
# bash ./scripts/run_chatbot_zh_wiki.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${test_dataset_path} ./output_models/${exp_name} 


# count=0
# echo "count: $count"
# lr=8e-5
# bs=2
# per_device_eval_batch_size=1
# use_lora=1
# epochs=1
# gradient_checkpointing=False
# gradient_accumulation_steps=16
# lora_r=3072
# block_size=1024
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=bigscience/bloom-7b1
# exp_name="xl_0126";
# data_path=${split_data_path}/split_${count}/
# eval_dataset_path="/home/xiangliu/LMFlow/data/continue_wiki_formated_eval/"
# bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 
# count=`expr $count + 1`

# while [ $count -lt $k ]
# do 
#     echo "count: $count"
#     model_name_or_path=./output_models/${exp_name}
#     data_path=${split_data_path}/split_${count}/
#     bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 
#     count=`expr $count + 1`
# done


# test_dataset_path="/home/xiangliu/LMFlow/data/continue_wiki_formated_test"
# bash ./scripts/run_chatbot_zh_wiki.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${test_dataset_path} ./output_models/${exp_name} 


# while [ $count -lt $k ]
# do 
#     echo "count: $count"
#     model_name_or_path=./output_models/${exp_name}
#     data_path=${split_data_path}/split_${count}/
#     bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 
#     count=`expr $count + 1`
# done

# test_dataset_path="/home/xiangliu/LMFlow/data/continue_wiki_formated_test"
# bash ./scripts/run_chatbot_zh_wiki.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${test_dataset_path} ./output_models/${exp_name} 

