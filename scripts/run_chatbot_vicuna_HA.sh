#!/bin/bash

model=gpt2
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi
# CUDA_VISIBLE_DEVICES=0 \
#   deepspeed examples/chatbot_test.py \
#       --deepspeed configs/ds_config_chatbot.json \
#       --model_name_or_path ${model} \
#       ${lora_args} \
#       --use_ram_optimized_load False \
#       --prompt_structure "###Question: could you give me the gist of how it could though? ###Answer:Sure. The basic idea would be to use algebraic topology to analyze the structure of the Minesweeper game board and determine the locations of mines based on certain topological invariants. This could involve representing the game board as a topological space and using topological invariants such as homotopy groups and Betti numbers to infer information about the location of mines on the board. However, as I mentioned before, implementing this would require a deep understanding of both algebraic topology and the game of Minesweeper, so it's not something that can be explained in a simple answer. ###Question: {input_text} ###Answer:" \
#       --max_new_tokens 400\
#       --end_string "###" 



CUDA_VISIBLE_DEVICES=0 \
  deepspeed examples/chatbot.py \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      ${lora_args} \
      --use_ram_optimized_load False \
      --prompt_structure "###Human: {input_text}###Assistant:" \
      --max_new_tokens 400 \
      --end_string "###"

# CUDA_VISIBLE_DEVICES=0 \
#   deepspeed examples/chatbot_test.py \
#       --deepspeed configs/ds_config_chatbot.json \
#       --model_name_or_path ${model} \
#       ${lora_args} \
#       --use_ram_optimized_load False \
#       --prompt_structure "###Human: How to cook a dumpling?###Assistant: 1.Boil a large pot of water. 2.Add dumplings to the pot and stir gently so they don’t stick together. 3.Bring the water back to a boil and add a cup of cold water. 4.Repeat step 3 two more times.###Human:{input_text} ###Assistant:" \
#       --max_new_tokens 400\
#       --end_string "###"

# CUDA_VISIBLE_DEVICES=0 \
#   deepspeed examples/chatbot_test.py \
#       --deepspeed configs/ds_config_chatbot.json \
#       --model_name_or_path ${model} \
#       ${lora_args} \
#       --use_ram_optimized_load False \
#       --prompt_structure "###Human: could you give me the gist of how it could though?###Assistant:Sure. The basic idea would be to use algebraic topology to analyze the structure of the Minesweeper game board and determine the locations of mines based on certain topological invariants. This could involve representing the game board as a topological space and using topological invariants such as homotopy groups and Betti numbers to infer information about the location of mines on the board. However, as I mentioned before, implementing this would require a deep understanding of both algebraic topology and the game of Minesweeper, so it's not something that can be explained in a simple answer.###Human: {input_text}###Assistant:" \
#       --max_new_tokens 400\
#       --end_string "###" 

# --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: What are the key differences between renewable and non-renewable energy sources?###Assistant: Renewable energy sources are those that can be replenished naturally in a relatively short amount of time, such as solar, wind, hydro, geothermal, and biomass. Non-renewable energy sources, on the other hand, are finite and will eventually be depleted, such as coal, oil, and natural gas. Here are some key differences between renewable and non-renewable energy sources:\n1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable energy sources are finite and will eventually run out.\n2. Environmental impact: Renewable energy sources have a much lower environmental impact than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, and other negative effects.\n3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically have lower operational costs than non-renewable sources.\n4. Reliability: Renewable energy sources are often more reliable and can be used in more remote locations than non-renewable sources.\n5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different situations and needs, while non-renewable sources are more rigid and inflexible.\n6. Sustainability: Renewable energy sources are more sustainable over the long term, while non-renewable sources are not, and their depletion can lead to economic and social instability.\n###Human: {input_text}###Assistant:" \
# --end_string "###" \
# --max_new_tokens 512


# CUDA_VISIBLE_DEVICES=0 \
#   deepspeed examples/chatbot.py \
#       --deepspeed configs/ds_config_chatbot.json \
#       --model_name_or_path ${model} \
#       ${lora_args} \
#       --use_ram_optimized_load False \
#       --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input_text}###Assistant:" \
#       --end_string "###" 



# CUDA_VISIBLE_DEVICES=0 \
#   deepspeed examples/chatbot.py \
#       --deepspeed configs/ds_config_chatbot.json \
#       --model_name_or_path ${model} \
#       ${lora_args} \
#       --use_ram_optimized_load False \
#       --prompt_structure "Below is an instruction that describes a task. Write a response that appropriately completes the request.###Human: {input_text}###Assistant:" \
#       --max_new_tokens 400\
#       --end_string "###"