import json
import re
import json_repair
import argparse
# import os

def tag_completion(mention, sentence):
    idx = 0
    while idx < len(sentence):
        idx = sentence.find(mention, idx)
        if idx == -1:
            break
        start_tag = '<e> '
        end_tag = ' </e>'
        start_idx = idx - len(start_tag)
        end_idx = idx + len(mention)
        # 检查是否存在开始标签
        has_start_tag = sentence[start_idx:idx] == start_tag if start_idx >= 0 else False
        # 检查是否存在结束标签
        has_end_tag = sentence[end_idx:end_idx + len(end_tag)] == end_tag if end_idx + len(end_tag) <= len(sentence) else False
        # 如果缺少开始标签，插入开始标签
        if not has_start_tag:
            sentence = sentence[:idx] + start_tag + sentence[idx:]
            idx += len(start_tag)
            end_idx += len(start_tag)
        # 如果缺少结束标签，插入结束标签
        if not has_end_tag:
            sentence = sentence[:end_idx] + end_tag + sentence[end_idx:]
            idx += len(end_tag)
        idx = end_idx + len(end_tag)
    return sentence

def check_and_renumber_entities(entities):
    is_sequential = all(d['entity_id'] == i for i, d in enumerate(entities))
    
    if not is_sequential:
        for i, d in enumerate(entities):
            d['entity_id'] = i  # 重新设置合法编号
    
    return entities

if __name__ == "__main__":
    output = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="gpt-4o-mini", help="The teacher model to use for training.")
    parser.add_argument("--student_model", type=str, default="chatglm3-6b", help="The student model to use for training.")
    parser.add_argument("--dataset", type=str, default="docred", help="The dataset to use.")
    parser.add_argument("--dev_or_test", type=str, default="dev", help="The dataset to use, dev or test.")
    
    args = parser.parse_args()
    teacher_model = args.teacher_model
    student_model = args.student_model
    dataset = args.dataset
    dev_or_test = args.dev_or_test

    path = f"./saves/{teacher_model}/{dataset}/student_output/{student_model}/{dev_or_test}"
    with open(f"{path}/generated_predictions.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            json_data = json.loads(line)
            p = json_data["predict"]
            start_index = p.find('[')
            end_index = p.rfind(']')

            k = p[start_index: end_index+1]
            json_obj = json_repair.loads(k)

            if len(json_obj) == 0:
                json_obj = []
            
            final_output = []
            for item in json_obj:
                if isinstance(item, dict) and "entity_id" in item and "mentions" in item and "description" in item and isinstance(item["entity_id"], int) and isinstance(item["mentions"], list) and isinstance(item["description"], str):
                    if isinstance(item["description"], str) and re.search(r'<e>.*?</e>', item["description"]):
                        final_output.append(item)
                    else:
                        abnormal = item
                        mentions = sorted(item["mentions"], key=len, reverse=True)
                        for mention in mentions:
                            if mention in abnormal["description"]:
                                abnormal["description"] = tag_completion(mention, abnormal["description"])
                                final_output.append(abnormal)
                                break          
            final_output = check_and_renumber_entities(final_output)

            output.append({"text_id" : i, "entity_descriptions" : final_output})

    with open(f"{path}/entity_description.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)