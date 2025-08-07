import argparse
from tqdm import tqdm
import argparse
import openai
from jinja2 import Template
import os,re
import json
from transformers import AutoTokenizer
from jinja2 import Template


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

#得到如下形式 [('prompt', 'response'), ('prompt', 'response'), ...]
def load_file(input_fp):
    # if the file is json file, load it
    if input_fp.endswith('.json'):
        with open(input_fp, 'r') as f:
            data = json.load(f)
    elif input_fp.endswith('.jsonl'):
        data = []
        with open(input_fp, 'r') as f:
            for line in f:
                data.append(json.loads(line))            
    else:
        raise ValueError(f"Unsupported file format: {input_fp}")
    prompt_response = []
    prompt_response_pairs = []
    for key,item in data.items():
        entry ={
            'prompt':item['query'],
            "responses": [
                item['response_0'],
                item['response_1'],
                item['response_2'],
                item['response_3']
            ]
        }
        prompt_response.append(entry)
    for item in prompt_response:
        for response in item["responses"]:
            prompt_response_pairs.append((item["prompt"], response))
    return prompt_response_pairs,prompt_response

def remove_think_blocks(text):
    text = postprocess_output(text)
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def parse_score(text):
    text = remove_think_blocks(text)
    """从模型输出中提取评分"""
    patterns = [
            r'\b(\d+)\b',              # 纯数字
            r'score:\s*(\d+)',          # "score: 8"
            r'rating:\s*(\d+)'          # "rating: 7"
        ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            return max(1, min(10, score))  # 确保在1-10范围内
    return 5 

def build_triplets(dataset, scores):
    """构建三元组数据集"""
    triplets = []
    
    # 准备所有评分样本
    prompt_response_pairs = []
    for item in dataset:
        for response in item["responses"]:
            prompt_response_pairs.append((item["prompt"], response))
    # 重组为三元组
    score_idx = 0
    for item in dataset:
        prompt = item["prompt"]
        response_scores = []
        
        # 获取当前prompt的4个评分
        for response in item["responses"]:
            score = scores[score_idx]
            response_scores.append((response, score))
            score_idx += 1
        
        # 选择最优和最差响应
        chosen = max(response_scores, key=lambda x: x[1])
        rejected = min(response_scores, key=lambda x: x[1])
        
        # 构建三元组
        triplets.append({
            "prompt": prompt,
            "chosen": chosen[0],
            "rejected": rejected[0],
            "scores": [score for _, score in response_scores]  # 可选：保存所有评分
        })
    
    return triplets

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=6000)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template',type=bool, default=True)
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str,default='api')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=200)    
    parser.add_argument('--task_floder', type=str, default='anonymous_run')
    args = parser.parse_args()

    print(f"Using local API server at port {args.port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,trust_remote_code=True,enable_thinking=False,padding_side='left')
        template = Template(tokenizer.chat_template)

    def call_model(prompts, model, max_new_tokens=50, print_example =False):
        temperature = 0.0
        if print_example:
            print("Example:")
            print(prompts[1])
        preds = []
        if args.use_chat_template: 
            prompts = [template.render(messages=[{"role": "user", "content": prom}],bos_token= tokenizer.bos_token,add_generation_prompt=True) for prom in prompts]
        
        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt,add_special_tokens= False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt[-args.max_tokens:])
            prompts = new_prompts

        response = client.completions.create(
            model="default",
            prompt=prompts,
            temperature=temperature, top_p=0.9, max_tokens=max_new_tokens
        )
        preds = [x.text for x in response.choices]
        postprocessed_preds = [parse_score(pred) for pred in preds]
        return postprocessed_preds, preds

    input_data,prompt_response = load_file(args.data_file)
    model = None
    all_scores = []
    SCORE_PROMPT = """
        Please evaluate the response quality on a scale of 1-10 based on the following criteria:
        1. Relevance: Does it directly address the user's question?
        2. Informativeness: Does it provide sufficient and useful information?
        3. Logical consistency: Is the expression clear and well-structured?
        4. Safety: Is the content safe and reliable?

        Input: {prompt}
        Response: {response}

        Please output only the score number without any additional text.
        """
    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch = input_data[idx*args.batch_size:min((idx+1)*args.batch_size, len(input_data))]
        if len(batch) == 0:
            break
        processed_batch = [
            SCORE_PROMPT.format(prompt=prompt, response=response)
            for prompt, response in batch]
        
        if idx == 0:
            print_example = True
        else:
            print_example = False

        preds, _ = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens, print_example=print_example)
        
        all_scores += preds
    task_name = os.path.split(args.model_name)[-1]
    triplets = build_triplets(prompt_response, all_scores)
    save_path = f'{args.task_floder}/{task_name}_triplets.json'

    with open(save_path, 'w') as f:
        json.dump(triplets,f,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()