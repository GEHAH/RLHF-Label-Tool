log_num=0
task_floder=/owenbhe/buddy1/lqszchen/O_Medical_o1/data/RL_data
model_name=model/Qwen3-32B
port=28${log_num}35

data_file=/owenbhe/buddy1/lqszchen/O_Medical_o1/data/RL_data_select.json
python /owenbhe/buddy1/lqszchen/O_Medical_o1/create_RLdata_step2.py --model_name $model_name  --data_file $data_file --port $port --strict_prompt --batch_size 20 --max_new_tokens 4096 --task_floder $task_floder
