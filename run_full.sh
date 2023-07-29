
lr=1e-5
bs=45
per_device_eval_batch_size=1
use_lora=0
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=128
block_size=2048
ds_config=configs/ds_config_zero3.json
model_name_or_path=pinkmanlove/llama-7b-hf
exp_name="continue_lora_A100*7_full_compare";
data_path="data/high_SFT_data_0501/"
echo ${data_path}
eval_dataset_path="data/eval_760/"
test_dataset_path="data/eval_760"
bash ./scripts/run_finetune_full_xl.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10001 --include localhost:0,1,2,3,4,5,6" 

bash ./scripts/run_evaluation.sh ./output_models/${exp_name}  ${test_dataset_path} "--master_port=10005 --include localhost:0"

python /homes/rpan/xiangL/lmflow_xl/llama_eval.py --device_id 0 1 2  3 4 5 6  --gb 70 