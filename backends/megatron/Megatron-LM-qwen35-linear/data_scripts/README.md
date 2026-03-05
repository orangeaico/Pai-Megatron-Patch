1) Convert raw text file to jsonl format by splitting at the specified separator. This will create the output file with the same name as input file but with extension .jsonl

python data_scripts/split_text_to_jsonl.py --input_file <input_text_file> --separator <sep>

2) Split the jsonl file into train and val subsets. The eval_ratio (between 0 to 100) with default value 5 specifies what percentage of data to be assigned to validation set. This will create 2 files in the same path with suffix _train and _val. 

python data_scripts/split_train_val.py <input_jsonl_file> --eval-ratio <eval_ratio>

3) Use the Megatron-LM tools/preprocess_data.py script inside the Megatron container to convert the json files to .bin/.idx format. Specify the output path in the output-prefix argument. Run it separately for both train and val jsonl files.

python tools/preprocess_data.py --input /workspace/data/data/qwen3-data-prep/all_chunks_concatenated_train.jsonl --output-prefix /workspace/data/data/qwen3-data-prep/all_chunks_concatenated_train --tokenizer-type HuggingFaceTokenizer --tokenizer-model /workspace/data/mega-models/Qwen3-1.7B --workers 4 --append-eod