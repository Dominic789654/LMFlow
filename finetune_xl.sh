


bash ./scripts/llama_SFT_30.sh "--master_port=11000 --include localhost:0,1,2,3,4,5,6,7" > ./log/SFT_30_llama_7b.log 2>&1



bash ./scripts/bloom_SFT_30.sh "--master_port=11000 --include localhost:0,1,2,3,4,5,6,7" > ./log/bloom_SFT_30.log 2>&1



bash ./scripts/lora_SFT_gptj_7b.sh "--master_port=11000 --include localhost:0,1,2,3,4,5,6,7" > ./log/lora_SFT_gptj_7b.log 2>&1