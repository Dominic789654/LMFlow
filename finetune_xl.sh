# lr=8e-4
# bs=4
# use_lora=1
# epochs=3.5
# gradient_checkpointing=False
# gradient_accumulation_steps=4
# lora_r=32
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=pinkmanlove/llama-7b-hf
# exp_name="xl_061";
# data_path="/home/xiangliu/LMFlow/data/cn_v1_text2text_end_single_round_HA"
# eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
# bash ./scripts/run_finetune_xl.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path}  "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# evaluation_path=/home/xiangliu/LMFlow/data/gpt4_eval
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${evaluation_path} ${model_name_or_path}


lr=8e-4
bs=4
use_lora=1
epochs=3.5
gradient_checkpointing=False
gradient_accumulation_steps=4
lora_r=128
ds_config=configs/ds_config_zero2.json
model_name_or_path=pinkmanlove/llama-7b-hf
exp_name="xl_062";
data_path="/home/xiangliu/LMFlow/data/cn_v1_text2text_end_single_round_HA"
eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
lora_target_modules="q_proj k_proj v_proj o_proj"
bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${lora_target_modules} "--master_port=10065 --num_gpus=8" 

bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

evaluation_path=/home/xiangliu/LMFlow/data/gpt4_eval
bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${evaluation_path} ${model_name_or_path}

# lr=8e-4
# bs=4
# use_lora=1
# epochs=3.5
# gradient_checkpointing=False
# gradient_accumulation_steps=2
# lora_r=32
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=pinkmanlove/llama-7b-hf
# exp_name="xl_063";
# data_path="/home/xiangliu/LMFlow/data/cn_v1_text2text_end_single_round_HA"
# eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
# bash ./scripts/run_finetune_xl.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# evaluation_path=/home/xiangliu/LMFlow/data/gpt4_eval
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${evaluation_path} ${model_name_or_path}

# lr=8e-4
# bs=4
# use_lora=1
# epochs=3.5
# gradient_checkpointing=False
# gradient_accumulation_steps=1
# lora_r=32
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=pinkmanlove/llama-7b-hf
# exp_name="xl_064";
# data_path="/home/xiangliu/LMFlow/data/cn_v1_text2text_end_single_round_HA"
# eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
# lora_target_modules=
# bash ./scripts/run_finetune_xl.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${lora_target_modules} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# evaluation_path=/home/xiangliu/LMFlow/data/gpt4_eval
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${evaluation_path} ${model_name_or_path}

# lr=8e-4
# bs=4
# use_lora=1
# epochs=3.5
# gradient_checkpointing=False
# gradient_accumulation_steps=2
# lora_r=128
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=pinkmanlove/llama-7b-hf
# exp_name="xl_065";
# data_path="/home/xiangliu/LMFlow/data/cn_v1_text2text_end_single_round_HA"
# eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
# lora_target_modules="q_proj k_proj v_proj o_proj"
# bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${lora_target_modules} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# evaluation_path=/home/xiangliu/LMFlow/data/gpt4_eval
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${evaluation_path} ${model_name_or_path}

# lr=8e-4
# bs=4
# use_lora=1
# epochs=3.5
# gradient_checkpointing=False
# gradient_accumulation_steps=1
# lora_r=128
# ds_config=configs/ds_config_zero2.json
# model_name_or_path=pinkmanlove/llama-7b-hf
# exp_name="xl_066";
# data_path="/home/xiangliu/LMFlow/data/cn_v1_text2text_end_single_round_HA"
# eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
# lora_target_modules="q_proj k_proj v_proj o_proj"
# bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${lora_target_modules} "--master_port=10065 --num_gpus=8" 

# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# evaluation_path=/home/xiangliu/LMFlow/data/gpt4_eval
# bash ./scripts/run_evaluation_nll_HA.sh ${exp_name} nll ${evaluation_path} ${model_name_or_path}

# for exp_name in 027 029 039 
# do
#     bash ./scripts/run_evaluation_nll_QA.sh xl_${exp_name} nll ./output_models/xl_${exp_name}
# done

# for exp_name in 031 033 037 038 042
# do
#     bash ./scripts/run_evaluation_nll_HA.sh xl_${exp_name} nll ./output_models/xl_${exp_name}
# done
