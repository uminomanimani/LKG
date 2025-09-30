from openai import OpenAI, APIConnectionError
import json
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import json_repair
from threading import Semaphore
import os
import argparse
from dotenv import load_dotenv

def request_api(api_key: str, prompt, model: str, previous_response: str = None):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.agicto.cn/v1"
    )

    # 构建上下文消息
    messages = []
    messages.append({"role": "user", "content": prompt})
    
    if previous_response:
        messages.append({"role": "assistant", "content": previous_response})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    response = chat_completion.choices[0].message.content
    finish_reason = chat_completion.choices[0].finish_reason

    return {
        "response": response,
        "finish_reason": finish_reason
    }
    
with open("./dataset/re-docred/data/rel_info.json", "r", encoding="utf-8") as f:
    rel_info = json.load(f)

def parse_and_check_llm_output(llm_output, original_data):
    # valid = True
    start_pos = llm_output.find("[")
    end_pos = llm_output.rfind("]")
    if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
        llm_output = llm_output[start_pos:end_pos+1]
    else:
        return {
            "llm_output": llm_output,
            "valid": False,
        }
    llm_output_json = json_repair.loads(llm_output)
    if len(original_data["vertexSet"]) != len(llm_output_json):
        return {
            "llm_output": llm_output,
            "valid": False,
        }
    for i, item in enumerate(llm_output_json):
        description = item["description"]
        if "<e>" in description and "</e>" in description and description.find("<e>") < description.find("</e>"):
            continue
        else:
            return {
                "llm_output": llm_output,
                "valid": False,
            }
    return {
        "llm_output": llm_output_json,
        "valid": True
    }



# 限制最多同时访问模型的线程数（比如 8）
model_semaphore = Semaphore(16)

def request_description(data, prompt, api_key, model, alter_model):
    with model_semaphore:  # 控制并发访问模型
        text = ""
        sents = data["sents"]
        for sent in sents:
            text += " ".join(sent)
        entities = data["vertexSet"]
        es = []
        request_model = model

        for j, entity in enumerate(entities):
            mentions = [m["name"] for m in entity]
            es.append({"entity_id": j, "mentions": mentions})

        json_data = {"text": text, "entities": es}
        json_data = json.dumps(json_data, ensure_ascii=False)
        prompt_input = prompt.replace(r"{real_data}", json_data)
        max_retry = 3
        llm_output_json = None

        for retry_index in range(max_retry):
            try:
                previous_response = None
                llm_output_str = ""

                # 限制拼接最大步数，防止死循环
                max_steps = 5
                for step in range(max_steps):
                    llm_output = request_api(
                        api_key=api_key,
                        prompt=prompt_input,
                        model=request_model,
                        previous_response=previous_response
                    )

                    response_chunk = llm_output["response"]
                    llm_output_str += response_chunk

                    if llm_output["finish_reason"] == "stop":
                        break

                    previous_response = llm_output_str
                else:
                    print("Exceeded max steps in response loop.", flush=True)
                    # break

                # 解析模型输出
                parsed_result = parse_and_check_llm_output(llm_output_str, original_data=data)

                if parsed_result and parsed_result["valid"]:
                    llm_output_json = parsed_result["llm_output"]
                    break  # 成功，退出 retry 循环
                else:
                    request_model = alter_model
                    print("Switched to alternate model. Sleeping to prevent congestion.", flush=True)
                    sleep(3.0)
            except APIConnectionError:
                print("API connection error. Retrying after short sleep.", flush=True)
                sleep(3.0)

        return llm_output_json if llm_output_json is not None else []

def multithreaded_request(
    data_array,
    prompt,
    api_key,
    model,
    alter_model,
    n_threads
):
    output = [None] * len(data_array)
    total = len(data_array)

    def task(i, data):
        result = request_description(data, prompt, api_key, model, alter_model)
        return i, result

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        with tqdm(total=total, desc="Processing", ncols=80) as pbar:
            futures = [executor.submit(task, i, data) for i, data in enumerate(data_array)]
            for future in as_completed(futures):
                try:
                    i, result = future.result()
                except TimeoutError:
                    print(f"Task {i} timed out", flush=True)
                    i, result = -1, []
                output[i] = result
                pbar.update(1)

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="docred", help="The dataset to use.")
    parser.add_argument("--teacher_model", type=str, default="gpt-4o-mini", help="The teacher model to use for requesting entity descriptions.")
    parser.add_argument("--dataset_type", type=str, default="test", help="The dataset type to use, train, dev or test.")
    parser.add_argument("--n_threads", type=int, default=32, help="Number of threads to use for multithreading.")
    args = parser.parse_args()

    dataset = args.dataset
    teacher_model = args.teacher_model
    dataset_type = args.dataset_type
    n_threads = args.n_threads

    dataset_name = json.load(open("./dataset_name.json", "r"))
    data_file = dataset_name[dataset][dataset_type]

    load_dotenv()
    API_KEY = os.environ.get("API_KEY")
    print(f"Requesting entity descriptions using model: {teacher_model} on dataset: {dataset} ({dataset_type}) with {n_threads} threads.", flush=True)
    print(f"This may be expensive. Ensure you have sufficient quota.", flush=True)
    prompt_path = "./prompt/prompt_entity_description.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    data_path = f"./dataset/{dataset}/data/{data_file}"
    with open(data_path, "r", encoding="utf-8") as f:
        data_array = json.load(f)
    
    output = multithreaded_request(
        data_array=data_array,
        prompt=prompt,
        api_key=API_KEY,
        model=teacher_model,
        alter_model="gpt-4o",
        n_threads=n_threads
    )   

    output_data = []
    for i, item in enumerate(output):
        output_data.append({"text_id" : i, "entity_descriptions" : item})
    
    os.makedirs(f"./saves/{teacher_model}/{dataset}/teacher_output/", exist_ok=True)
    with open(f"./saves/{teacher_model}/{dataset}/teacher_output/entity_description_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
 