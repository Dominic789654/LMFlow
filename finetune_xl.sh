# lr=1e-4
# bs=4
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_013_xlgpt4_cn_v3_QA_vicuna7b_3epcoh_lr${lr}";
# data_path="/home/xiangliu/LMFlow/data/xl_gpt4_cn_v3_QA"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_QA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# zhishi_01
# lr=2e-5
# bs=4
# use_lora=0
# ds_config=configs/ds_config_zero3.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="zhishi_01_nolora";
# data_path="/home/xiangliu/LMFlow/data/zhishi_01"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=2e-5
# bs=4
# use_lora=1
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_014";
# data_path="/home/xiangliu/LMFlow/data/zhishi_02"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=2e-5
# bs=4
# use_lora=1
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=aleksickx/llama-7b-hf
# exp_name="xl_015";
# data_path="/home/xiangliu/LMFlow/data/zhishi_02"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=2e-5
# bs=4
# use_lora=1
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_016";
# data_path="/home/xiangliu/LMFlow/data/zhishi_03"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=2e-5
# bs=4
# use_lora=1
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_019";
# data_path="/home/xiangliu/LMFlow/data/hardcode_01/"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 
# exp_name="xl_021";
# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=1e-4
# bs=4
# use_lora=1
# epochs=200
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_022";
# data_path="/home/xiangliu/LMFlow/data/hardcode_01/"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 


# lr=1e-4
# bs=4
# use_lora=1
# epochs=100
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_023";
# data_path="/home/xiangliu/LMFlow/data/hardcode_01/"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=1e-4
# bs=4
# use_lora=1
# epochs=100
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_024";
# data_path="/home/xiangliu/LMFlow/data/hardcode_03/"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# bash ./scripts/run_evaluation_nll_HA.sh baseline_llama_7b_nll_HA nll aleksickx/llama-7b-hf

# bash ./scripts/run_evaluation_nll_QA.sh baseline_llama_7b_nll_QA nll aleksickx/llama-7b-hf

# # bash ./scripts/run_evaluation_nll_HA.sh baseline_vicuna_7b_nll_HA nll Tribbiani/vicuna-7b

# bash ./scripts/run_evaluation_nll_QA.sh baseline_vicuna_7b_nll_QA nll Tribbiani/vicuna-7b

lr=2e-5
bs=4
use_lora=0
epochs=3
ds_config=configs/ds_config_zero2.json
model_name_or_path=aleksickx/llama-7b-hf
exp_name="xl_027";
data_path="xl_sharegpt_20230401_clean_lang_split_v1_QA"
bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} "--master_port=10065 --num_gpus=8" 

bash ./scripts/run_chatbot_vicuna_test_QA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

bash ./scripts/run_evaluation_nll_QA.sh ${exp_name} nll ./output_models/${exp_name}

# test 知识注入
# nolora_en
# lr=2e-5
# bs=4
# use_lora=0
# ds_config=configs/ds_config_zero3.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="test_zhishi_en_nolora";
# data_path="/home/xiangliu/LMFlow/data/test_chat_en"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 


# lr=2e-5
# bs=4
# use_lora=0
# ds_config=configs/ds_config_zero3.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="test_zhishi_zh_nolora";
# data_path="/home/xiangliu/LMFlow/data/test_chat_zh"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=2e-5
# bs=4
# use_lora=0
# ds_config=configs/ds_config_zero3.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="test_zhishi_mix_nolora";
# data_path="/home/xiangliu/LMFlow/data/test_chat_mix"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# uselora
# lr=1e-4
# bs=4
# use_lora=1
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="test_zhishi_en_lora";
# data_path="/home/xiangliu/LMFlow/data/test_chat_en"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=1e-4
# bs=4
# use_lora=1
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="test_zhishi_zh_lora";
# data_path="/home/xiangliu/LMFlow/data/test_chat_zh"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=1e-4
# bs=4
# use_lora=1
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="test_zhishi_mix_lora";
# data_path="/home/xiangliu/LMFlow/data/test_chat_mix"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=1e-4
# bs=2
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_011_xlShareGPT_V2_unfilter_cleaned_QA_vicuna7b_lora_3epcoh_lr${lr}";
# data_path="/home/xiangliu/LMFlow/data/xl_ShareGPT_V2_unfiltered_cleaned_split_QA"
# # bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_QA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# lr=1e-4
# bs=4
# model_name_or_path=/home/xiangliu/LMFlow/output_models/xl_011_xlShareGPT_V2_unfilter_cleaned_QA_vicuna7b_lora_3epcoh_lr1e-4
# exp_name="xl_012_continue_011_on_gpt_cn_v3_lora_3epcoh_lr${lr}";
# data_path="/home/xiangliu/LMFlow/data/xl_gpt4_cn_v3_QA"
# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_QA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 


# lr=1e-4
# bs=2
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_004_sharegpt_v3_text_only_QA_0.1_vicuna7b_lora_3epcoh_lr${lr}";
# data_path="/home/xiangliu/LMFlow/data/sharegpt_v3_QA_0.1"
# ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} "--master_port=10065 --num_gpus=8" 


# lr=1e-4
# bs=4
# model_name_or_path=Tribbiani/vicuna-7b
# exp_name="xl_005_alpaca_gpt4_data_zh_text_only_QA_vicuna7b_lora_3epcoh_lr${lr}"; 
# data_path="/home/xiangliu/LMFlow/data/alpaca_gpt4_data_zh"
# ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} "--master_port=10065 --num_gpus=8" 

# lr=2e-5
# bs=4
# # model_name_or_path=Tribbiani/vicuna-7b
# model_name_or_path=/home/xiangliu/LMFlow/output_models/xl_004_sharegpt_v3_text_only_QA_0.1_vicuna7b_lora_3epcoh_lr1e-4
# exp_name="xl_006_alpaca_gpt4_data_zh_text_only_QA_on_004_3epcoh_lr${lr}"; 
# data_path="/home/xiangliu/LMFlow/data/alpaca_gpt4_data_zh"
# ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} "--master_port=10065 --num_gpus=8" 

# exp_name="xl_003_sharegpt_v3_text_only_QA_0.1_vicuna7b_lora_3epcoh_lr1e-4"; 
# data_path="/home/xiangliu/LMFlow/data/sharegpt_v3_QA_0.1"
# ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} "--master_port=10065 --num_gpus=8" 


# exp_name="xl_003_sharegpt_v3_text_only_student_0.1_vicuna7b_lora_3epcoh_lr1e-4"; 
# data_path="/home/xiangliu/LMFlow/data/sharegpt_v3_student_0.1/"
# ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} "--master_port=10065 --num_gpus=8" 

# exp_name="xl_003_sharegpt_v3_text_only_student_0.1_vicuna7b_lora_3epcoh_lr1e-4"; 
# data_path="/home/xiangliu/LMFlow/data/sharegpt_v3_student_0.1/"
# ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} "--master_port=10065 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 512 --lora_r 8 --dataset_path data/sharegpt_v3/train/" 

# ./scripts/run_evaluation_ppl_cn.sh Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}
# ./scripts/run_evaluation_ppl_en.sh Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}

# exp_name="xl_002_sharegpt_v3_vicuna7b_lora_3epcoh_lr5e-5";
# ./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10066 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 5e-5 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v8/train/"

# ./scripts/run_evaluation_ppl_cnoome/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}


# bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh "--master_port=10065 --num_gpus=8"