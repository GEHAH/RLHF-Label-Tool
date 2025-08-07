# RLHF-Label-Tool 
在大模型训练的 RLHF 阶段，需要人工对模型生成的多份数据进行标注排序，然而对于样本量大的数据集，以往人工标注是较为耗时的，因此我们希望结合LLM 辅助对模型生成的多个 responses 进行打分标注和排序，具体操作如下所示：
## 功能特点
* 可以对每个回复进行打分
* 并基于打分生成 chosen_response 和 rejected_response,用于DPO
## 安装依赖
* python 3.x
* [sglang](https://github.com/sgl-project/sglang), [transformers](https://github.com/huggingface/transformers)
## 数据准备
1. 首先，通过利用我们经过SFT的模型对同一个 prompt生成多个 responses,并保存为JSON格式，参考 input_file.json,在此，我们以4 个回复为例，格式如下所示：
```python
input_data = {
        '0': {
            'query': 'Hi, I have a bit of swollen forehead...',
            'history': [],
            'response_0': 'Based on your description...',
            'response_1': 'Based on your description...',
            'response_2': 'Based on your situation...',
            'response_3': 'Based on your description...'
        },
        '1': {
            'query': 'My four year old son recently had...',
            'history': [],
            'response_0': "Based on the information provided...",
            'response_1': 'The recurring nosebleeds and dry cough...',
            'response_2': 'It sounds like your son’s symptoms...',
            'response_3': 'The recurring nosebleeds and dry cough...'
        }
    }
```
2. 我们使用`sglang`来加入处理进程，依次运行 `bash builtRL_1.sh`和`bash builtRL_2.sh`
```bash
log_num=0
model_name=model/Qwen3-32B
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server --model-path $model_name --port $port  --mem-fraction-static 0.8 --dp 8 --tp 1  > RL_server${log_num}.log 2>&1 &    

```

```bash
log_num=0
task_floder=data/RL_data
model_name=model/Qwen3-32B
port=28${log_num}35

data_file=data/input_file.json
python create_RLdata_step2.py --model_name $model_name  --data_file $data_file --port $port --strict_prompt --batch_size 20 --max_new_tokens 4096 --task_floder $task_floder
```
其中`task_floder`表示保存处理后的文件的路径，`data_file`为需要打分的文件，`model_name` 表示辅助我们打分排序的模型
3. 数据处理结束，我们需要结束`sglang server`，运行`bash kil_sglang_server.sh`
```bash
pkill -f sglang
pkill -f multiprocessing.spawn
```
