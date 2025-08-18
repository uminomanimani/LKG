import torch
import json
import random
from tqdm import tqdm
from loss import AFLoss as balanced_loss
from models.model import get_labels
from official_evaluate import official_evaluate


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

def to_official(preds, entity_pairs, id_to_rel, title):
    # pred [num_pairs, 97]
    out = []
    for pred, ht in zip(preds, entity_pairs):
        idx = torch.nonzero(pred).flatten().tolist()
        for i in idx:
            if i != 0:
                h = ht[0]
                t = ht[1]
                r = id_to_rel[i]
                out.append({"title" : title, "h_idx" : h, "t_idx" : t, "r" : r})
    return out

def eval_model(model, dev_graph, dev, label_to_id : dict, path, train_file, dev_file, num_labels=4, save=False):
    model.eval()
    id_to_label = {v : k for k, v in label_to_id.items()} 
    device = next(model.parameters()).device
    total_eval_loss = 0.0
    model.eval()
    pred_triples = []
    num_classes = 97
    loss_fn = balanced_loss()
    random.shuffle(dev_graph)
    with torch.no_grad():
        with tqdm(total=len(dev_graph), desc="evaluating...") as pbar:
            for i, item in enumerate(dev_graph):
                text_id = item["text_id"]
                
                entity_descriptions = item["entities"]
                relation_description_graph = item["edges"]
                
                if len(relation_description_graph) == 0:
                    pbar.write(f"skipped text {text_id}...")
                    pbar.update(1)
                    continue 
                
                triples = dev[text_id]["labels"]
                title = dev[text_id]["title"]
                

                out = model(entity_descriptions, relation_description_graph)
                logits = out["logits"]
                entity_pairs = out["pairs"]

                preds = get_pred(logits, num_labels)

                labels = get_labels(triples=triples, entity_pairs=entity_pairs, label_to_id=label_to_id, num_class=97).to(device)

                loss = loss_fn(logits, labels)
                total_eval_loss += loss.item()

                pred_triples += to_official(preds=preds, entity_pairs=entity_pairs, id_to_rel=id_to_label, title=title)
                pbar.update(1)

    print("calculating score...")
    if save:
        with open("eval_result.json", "w", encoding="utf-8") as f:
            json.dump(pred_triples, fp=f, indent=4)
    if len(pred_triples) == 0:
        f1, ign_f1, precision, recall = 0, 0, 0, 0
    else:
        f1, _, ign_f1, _, precision, recall = official_evaluate(pred_triples, path, train_file, dev_file)
    avg_loss = total_eval_loss / len(dev_graph)
    return {
        "f1" : f1,
        "ign_f1" : ign_f1,
        "precision" : precision,
        "recall" : recall,
        "avg_loss" : avg_loss
    }