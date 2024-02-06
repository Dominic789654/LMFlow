import json 
# Manually correct the JSON structure based on the extracted text
corrected_json_structure = {
    "type": "text_only",
    "instances": []
}


# Path to the JSON file you want to load
input_json_path = 'train.json'

# Load the JSON data from the file
questions = []
answers = []
decoder = json.JSONDecoder()
with open(input_json_path) as f:
    lines = f.readlines()
    for line in lines:
        json_res = decoder.raw_decode(line)[0]
        cur_question = json_res["question"].strip()
        cur_answers = json_res["answer"].strip()
        # cur_answers = json_res["answer"].split("#### ")[-1].replace(",", "")
        # print(f"{cur_question}\n{cur_answers}")
        # break
        corrected_json_structure['instances'].append({'text':f"{cur_question}\n{cur_answers}"})


# This is where you can manipulate the data if needed
# For example, if you want to add a new key-value pair, you could do:
# data['new_key'] = 'new_value'

# Path to the JSON file where you want to save the modified data
output_json_path = 'train_text_only.json'
print(corrected_json_structure['instances'][0])
# # Save the modified JSON data to the output file
with open(output_json_path, 'w') as file:
    json.dump(corrected_json_structure, file, indent=4)  # The 'indent' argument formats the JSON to be more readable

# print(f'JSON data has been saved to {output_json_path}')
