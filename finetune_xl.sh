lr=1e-4
bs=4
model_name_or_path=Tribbiani/vicuna-7b
exp_name="xl_007_gpt_cn_v3_QA_vicuna7b_lora_5epcoh_lr${lr}";
data_path="/home/xiangliu/LMFlow/data/gpt4_cn_v3_QA"
./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} "--master_port=10065 --num_gpus=8" 

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