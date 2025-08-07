log_num=0
model_name=/owenbhe/buddy1/lqszchen/O_Medical_o1/model/Qwen3-32B
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server --model-path $model_name --port $port  --mem-fraction-static 0.8 --dp 8 --tp 1  > RL_server${log_num}.log 2>&1 &    
