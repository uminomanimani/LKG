import torch
from tqdm import tqdm
import re
from official_evaluate import official_evaluate
import json
from transformers import RobertaModel, RobertaTokenizer
from models.model import RE
import argparse

def find_paths(edges, l, max_hop):
    from collections import deque, defaultdict

    # 构建邻接表
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)

    res = {}

    for s in l:
        distances = {s: 0}
        predecessors = {}
        queue = deque([s])

        while queue:
            node = queue.popleft()
            if distances[node] >= max_hop:
                continue
            for neighbor in adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    predecessors[neighbor] = node
                    queue.append(neighbor)

        for t in l:
            if t != s and t in distances and distances[t] <= max_hop:
                # 通过回溯前驱节点来构建路径
                path_nodes = []
                current = t
                while current != s:
                    path_nodes.append((predecessors[current], current))
                    current = predecessors[current]
                path_edges = list(reversed(path_nodes))
                res[(s, t)] = path_edges

    return res

# 需要将要比较的graph中的实体编号映射成标签里的实体编号
def check(revised, graph, max_hop=2):
    all_relations = {}
    missing_relations = {}
    
    num_labels = 0
    exist_labels = 0

    for t_r, t_g in zip(revised, graph):
        entity_pred_to_label = {}
        title = t_r["title"]
        
        entity_num_in_label = len(t_r["vertexSet"])
        entity_num_in_pred = len(t_g["entities"])
        entity_pred = []
        entity_label = [[] for _ in range(entity_num_in_label)]
        
        for entity in t_g["entities"]:
            entity_name = re.search(r'<e>(.*?)</e>', entity["description"], re.DOTALL).group(1).replace(" ", "")
            entity_pred.append(entity_name)
        
        for j, entity in enumerate(t_r["vertexSet"]):
            for m in entity:
                if m["name"].replace(" ", "") not in entity_label[j]:
                    entity_label[j].append(m["name"].replace(" ", ""))
                    
        entity_pred_hit = [0] * entity_num_in_pred
        entity_label_hit = [0] * entity_num_in_label
        
        for j, entity in enumerate(entity_pred):
            for k, entity_l in enumerate(entity_label):
                if entity in entity_l and entity_label_hit[k] == 0 and entity_pred_hit[j] == 0:
                    entity_label_hit[k] = 1
                    entity_pred_hit[j] = 1
                    entity_pred_to_label[j] = k
                    
        for j, x in enumerate(entity_pred_hit):
            if x == 0:
                entity_pred_to_label[j] = -1
        
        
        edges = t_g["edges"]
        all_edges = []
        for e in edges:
            all_edges.append((entity_pred_to_label[e["h"]], entity_pred_to_label[e["t"]]))
            all_edges.append((entity_pred_to_label[e["t"]], entity_pred_to_label[e["h"]]))
        v = list(range(len(t_r["vertexSet"])))
        paths = find_paths(all_edges, v, max_hop=max_hop)
        p = [i for i, j in paths.items()]
        num_labels += len(t_r["labels"])
        for l in t_r["labels"]:
            h = l["h"]
            t = l["t"]
            r = l["r"]
            if r not in all_relations:
                all_relations[r] = 1
            else:
                all_relations[r] += 1
            if (h, t) in p or (t, h) in p:
                exist_labels += 1
            else:
                if r not in missing_relations:
                    missing_relations[r] = 1
                else:
                    missing_relations[r] += 1
        pass
    return {
        "num_total_labels": num_labels,
        "exist_labels": exist_labels,
        "information_retention": f"{exist_labels / num_labels * 100}%"
    }

def get_pred(logits, num_labels=-1):
    # logits [n, 97]
    th_logit = logits[:, 0].unsqueeze(1)
    preds = torch.zeros_like(logits).to(logits)
    mask = (logits > th_logit)
    if num_labels > 0:
        top_v, _ = torch.topk(logits, num_labels, dim=1)
        top_v = top_v[:, -1]
        mask = (logits >= top_v.unsqueeze(1)) & mask
    preds[mask] = 1.0
    preds[:, 0] = (preds.sum(1) == 0.).to(logits)
    return preds # [n, 97]

def to_official(model, graph, label_to_id : dict, path, dev_file, num_labels=3):
    model.eval()
    pred_triples = []
    with open(f"{path}/{dev_file}", "r", encoding="utf-8") as f:
        dev = json.load(f)
        
    id_to_label = {v : k for k, v in label_to_id.items()} 
    with torch.no_grad():
        with tqdm(total=len(graph), desc="evaluating...") as pbar:
            for i, item in enumerate(graph):
                text_id = item["text_id"]
                
                entity_pred_to_label = {} # map entity index in prediction to entity index in label
                title = dev[text_id]["title"]
                
                entity_num_in_label = len(dev[text_id]["vertexSet"])
                entity_num_in_pred = len(item["entities"])
                entity_pred = []
                entity_label = [[] for _ in range(entity_num_in_label)]
                
                for entity in item["entities"]:
                    entity_name = re.search(r'<e>(.*?)</e>', entity["description"], re.DOTALL).group(1).strip().replace(" ", "")
                    entity_pred.append(entity_name)
                
                for j, entity in enumerate(dev[text_id]["vertexSet"]):
                    for m in entity:
                        if m["name"].replace(" ", "") not in entity_label[j]:
                            entity_label[j].append(m["name"].strip().replace(" ", ""))
                            
                entity_pred_hit = [0] * entity_num_in_pred
                entity_label_hit = [0] * entity_num_in_label
                
                for j, entity in enumerate(entity_pred):
                    for k, entity_l in enumerate(entity_label):
                        if entity in entity_l and entity_label_hit[k] == 0 and entity_pred_hit[j] == 0:
                            entity_label_hit[k] = 1
                            entity_pred_hit[j] = 1
                            entity_pred_to_label[j] = k
                            
                for j, x in enumerate(entity_pred_hit): # false positive entities, map it to id of -1
                    if x == 0:
                        entity_pred_to_label[j] = -1
                    
                            
                entity_descriptions = item["entities"]
                relation_description_graph = item["edges"]
                
                out = model(entity_descriptions, relation_description_graph)
                logits = out["logits"]
                entity_pairs = out["pairs"]
                preds = get_pred(logits, num_labels)
                
                for pred, ht in zip(preds, entity_pairs):
                    idx = torch.nonzero(pred).flatten().tolist()
                    for i in idx:
                        if i != 0:
                            h = ht[0]
                            t = ht[1]
                            r = id_to_label[i]
                            h = entity_pred_to_label[h]
                            t = entity_pred_to_label[t]
                            pred_triples.append({
                                "title": title,
                                "h_idx": h,
                                "t_idx": t,
                                "r": r
                            })    
                pbar.update(1)       
    
    return pred_triples


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--teacher_model", type=str, default="gpt-4o-mini", help="The teacher model used.")
    argparse.add_argument("--student_model", type=str, default="mistral-7b", help="The student model used.")
    argparse.add_argument("--dataset", type=str, default="re-docred", help="The dataset to use.")
    argparse.add_argument("--dev_or_test", type=str, default="test", help="The dataset to evaluate, dev or test.")
    argparse.add_argument("--local_eval", type=bool, default=True, help="If True, evaluate locally, otherwise save the result to file and upload to the server.")
    argparse.add_argument("--max_hop", type=int, default=3, help="The maximum hop to use for path finding.")
    
    parser = argparse.parse_args()
    teacher_model = parser.teacher_model
    student_model = parser.student_model
    dataset = parser.dataset
    dev_or_test = parser.dev_or_test
    local_eval = parser.local_eval
    max_hop = parser.max_hop

    dataset_name = json.load(open("./dataset_name.json", "r"))

    with open(f"./dataset/{dataset}/data/rel_to_id.json", "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    
    dataset_path = f"./dataset/{dataset}/data"
    train_file = dataset_name[dataset]["train"]
    
    evaluate_type = dev_or_test
    dev_file = f"{evaluate_type}_revised.json"
    output_path = f"./student_output/{teacher_model}/{dataset}/{dev_or_test}/{student_model}"
    
    with open(f"{output_path}/graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_hop = max_hop
    local_eval = True
    
    bert_name = "roberta-base"
    bert_model = RobertaModel.from_pretrained(bert_name)
    tokenizer = RobertaTokenizer.from_pretrained(bert_name)
    special_tokens = {'additional_special_tokens' : ['<e>', '</e>', '[entity1]', '[entity2]']}
    tokenizer.add_special_tokens(special_tokens)
    bert_model.resize_token_embeddings(len(tokenizer))
    model = RE(bert_model=bert_model, tokenizer=tokenizer, max_hop=max_hop, num_labels=97)
    model.load_state_dict(torch.load(f"graph_model_checkpoints/{teacher_model}/{dataset}/checkpoint.pth"))
    model = model.to(device)
    
    triples_res = to_official(model, graph, label_to_id, path=dataset_path, dev_file=dev_file, num_labels=3)
    
    if local_eval:
        info_retention = check(json.load(open(f"{dataset_path}/{dev_file}", encoding="utf-8")), graph, max_hop)
        re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train , re_p, re_r = official_evaluate(triples_res, dataset_path, train_file, dev_file)
        res = {
            "re_f1": re_f1,
            "evi_f1": evi_f1,
            "re_f1_ignore_train_annotated": re_f1_ignore_train_annotated,
            "re_f1_ignore_train": re_f1_ignore_train,
            "re_p": re_p,
            "re_r": re_r
        }
        res.update(info_retention)
        print({
            "information_retention": info_retention["information_retention"],
            "precision": re_p,
            "recall": re_r,
            "f1": re_f1,
            "ign f1" : re_f1_ignore_train_annotated
        })
    else:
        with open(f"{output_path}/result.json", "w", encoding="utf-8") as f:
            json.dump(triples_res, f, indent=4, ensure_ascii=False)
    