
test_dataset_path="data/eval_760"

# loop from 500 to 2000
for i in {500..2000..500}
do
    echo "checkpoint-${i}"
    exp_name="./output_models/continue_lora_A40*4_full_compare/checkpoint-${i}/"
    bash ./scripts/run_evaluation.sh ${exp_name}  ${test_dataset_path} "--master_port=10003 --include localhost:6" 
done
# exp_name="output_models/continue_lora_A40*4_full_compare/checkpoint-500"
# bash ./scripts/run_evaluation.sh ./output_models/${exp_name}  ${test_dataset_path} "--master_port=10005 --include localhost:7" > ./log/eval_full_checkpoint_500.log 2>&1 &
