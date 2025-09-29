import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="docred")
    parser.add_argument("--dataset_type", type=str, default="dev")
    parser.add_argument("--teacher_model", type=str, default="deepseek-chat")

    args = parser.parse_args()
    dataset = args.dataset
    dataset_type = args.dataset_type
    teacher_model = args.teacher_model

    dataset_name = json.load(open("./dataset_name.json", "r"))
    data_file = dataset_name[dataset][dataset_type]

    original_data = json.load(open(f"./dataset/{dataset}/data/{data_file}", "r", encoding="utf-8"))
    teacher_data = json.load(open(f"./teacher_output/{teacher_model}/{dataset}/entity_description_{dataset_type}.json"))

    prompt = open("./prompt/prompt_entity_extraction.md").read()

    out = []

    teacher_len = len(teacher_data)
    original_data = original_data[0:teacher_len]

    for i, item in enumerate(original_data):
        text = ""
        for sent in item["sents"]:
            text += " ".join(sent)
        
        descs = teacher_data[i]["entity_descriptions"]
        _descs = []
        for j, desc in enumerate(descs):
            pattern = r"<e>(.*?)</e>"
            mentions = []
            for m in item["vertexSet"][j]:
                if m["name"] not in mentions:
                    mentions.append(m["name"])
            # matches = re.findall(pattern, desc["description"], re.DOTALL)[0].strip()
            _descs.append({"entity_id" : desc["entity_id"], "mentions" : mentions, "description" : desc["description"]})
        out.append({"instruction" : prompt, "input" : text, "output" : "```json\n" + json.dumps(_descs, ensure_ascii=False) + "```"})

    os.makedirs(f"./fine_tuning_for_description_extraction/{teacher_model}/{dataset}", exist_ok=True)    
    with open(f"./fine_tuning_for_description_extraction/{teacher_model}/{dataset}/ner_sft_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    
    if os.path.exists(f"./fine_tuning_for_description_extraction/dataset_info.json"):
        with open(f"./fine_tuning_for_description_extraction/dataset_info.json", "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info[f"{teacher_model}-{dataset}-{dataset_type}"] = {"file_name" : f"./{teacher_model}/{dataset}/ner_sft_{dataset_type}.json"}

    with open(f"./fine_tuning_for_description_extraction/dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)
    
    
    