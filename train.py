import torch
import json
from transformers import RobertaModel, RobertaTokenizer
from models.model import RE, get_labels
from loss import AFLoss as balanced_loss
import random
from tqdm import tqdm
from evaluate_teacher import eval_model
from scheduler import adjustable_lr_scheduler
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="docred", help="The dataset to use.")
    parser.add_argument("--max_hop", type=int, default=2, help="The maximum hop for the model.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate for the optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation.")
    parser.add_argument("--warmup_ratio", type=float, default=0.18, help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument("--maintein_ratio", type=float, default=0.09, help="Maintain ratio for the learning rate scheduler.")
    parser.add_argument("--teacher_model", type=str, default="deepseek-chat", help="The teacher model to use for training.")
    parser.add_argument("--decay_factor", type=float, default=0.6, help="Decay factor for the learning rate scheduler.")
    parser.add_argument("--min_lr", type=float, default=5e-7, help="Minimum learning rate for the scheduler.")
    args = parser.parse_args()

    dataset_config = json.load(open(f"./dataset_config.json", "r"))
    train_file = dataset_config[args.dataset]["train"] # train_annotated.json or train_revised.json
    dev_file = dataset_config[args.dataset]["dev"]

    dataset = args.dataset
    max_hop = args.max_hop
    lr = args.lr
    max_grad_norm = args.max_grad_norm
    epochs = args.epochs
    accumulation_steps = args.accumulation_steps
    warmup_ratio = args.warmup_ratio
    maintein_ratio = args.maintein_ratio
    teacher_model = args.teacher_model
    decay_factor = args.decay_factor
    min_lr = args.min_lr

    graph_path = f"./teacher_output/{teacher_model}/{dataset}"
    dataset_path = f"./dataset/{dataset}/data"
    
    with open(f"{graph_path}/train_graph.json", "r", encoding="utf-8") as f:
        train_graph = json.load(f)
    
    with open(f"{dataset_path}/{train_file}", "r", encoding="utf-8") as f:
        train_annotated = json.load(f)
    
    with open(f"{graph_path}/dev_graph.json", "r", encoding="utf-8") as f:
        dev_graph = json.load(f)
        
    with open(f"{dataset_path}/{dev_file}", "r", encoding="utf-8") as f:
        dev = json.load(f)

    with open(f"{dataset_path}/rel_to_id.json", "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    bert_name = "roberta-base"
    bert_model = RobertaModel.from_pretrained(bert_name)
    tokenizer = RobertaTokenizer.from_pretrained(bert_name)
    special_tokens = {'additional_special_tokens' : ['<e>', '</e>', '[entity1]', '[entity2]']}
    tokenizer.add_special_tokens(special_tokens)
    bert_model.resize_token_embeddings(len(tokenizer))
    model = RE(bert_model=bert_model, tokenizer=tokenizer, max_hop=max_hop, num_labels=97)
    model = model.to(device)
    loss_fn = balanced_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    train_history = []
    eval_history = []  
    accumulation_loss = 0.0
    
    random.shuffle(train_graph) 
    
    train_data_filtered = []
    for item in train_graph:
        if len(item["edges"]) != 0:
            train_data_filtered.append(item)
    
    eval_data_filtered = []
    for item in dev_graph:
        if len(item["edges"]) != 0:
            eval_data_filtered.append(item)
    
    total_steps = int(len(train_data_filtered) * epochs // accumulation_steps)
    warmup_steps = int(total_steps * warmup_ratio) 
    maintein_steps = int(total_steps * maintein_ratio)
    
    scheduler = adjustable_lr_scheduler(optimizer, warmup_steps, maintein_steps, decay_factor, min_lr)

    print(f"total_steps : {total_steps}, \nwarmup_steps : {warmup_steps}, \nmax_hop : {max_hop}, \nmax_grad_norm : {max_grad_norm}, \naccumulation_steps : {accumulation_steps}, \nmax_lr : {lr}")
    
    best_f1 = 0
    save_dir = f'./graph_model_checkpoints/{teacher_model}/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    history_epochs = []

    for epoch in range(0, 0 + epochs):
        with tqdm(desc=f"training epoch {epoch}...", total=(len(train_data_filtered) + accumulation_steps - 1) // accumulation_steps) as pbar:
            pass
            random.shuffle(train_data_filtered)
            for i, item in enumerate(train_data_filtered):
                model.train()
                text_id = item["text_id"]
                
                entity_descriptions = item["entities"]
                relation_description_graph = item["edges"]
                            
                triples = train_annotated[text_id]["labels"]                   

                out = model(entity_descriptions, relation_description_graph)
                logits = out["logits"]
                entity_pairs = out["pairs"]

                labels = get_labels(triples=triples, entity_pairs=entity_pairs, label_to_id=label_to_id, num_class=97).to(device)

                loss = loss_fn(logits, labels)
                accumulation_loss += loss.item()
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    loss_entry = {
                        "epoch" : epoch,
                        "lr" : scheduler.get_current_lr()[0],
                        "step" : (i + 1) // accumulation_steps,
                        "loss" : accumulation_loss / accumulation_steps
                    }
                    accumulation_loss = 0.0
                    train_history.append(loss_entry)
                    # with open("train.json", "w") as f:
                        # json.dump(train_history, f, indent=4, ensure_ascii=False)
                    pbar.update(1)
            if (i + 1) % accumulation_steps != 0:
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                
                loss_entry = {
                    "epoch" : epoch,
                    "lr" : scheduler.get_current_lr()[0],
                    "step": (i + 1) // accumulation_steps + 1,
                    "loss": accumulation_loss / ((i + 1) % accumulation_steps)
                }
                accumulation_loss = 0.0
                train_history.append(loss_entry)

                # 立即将数据保存到JSON文件
                with open("train.json", 'w') as f:
                    json.dump(train_history, f, indent=4)
                pbar.update(1)
        
        result = eval_model(model=model, dev_graph=eval_data_filtered, dev=dev, label_to_id=label_to_id, num_labels=3, path=dataset_path, train_file=train_file, dev_file=dev_file)
        result["epoch"] = epoch
        f1 = result["f1"]
        eval_history.append(result)
        scheduler.decay_if_warmup_and_maintein_finished(f1)
        
        if f1 > best_f1:    
            best_f1 = f1
            if os.path.isfile(f"./{save_dir}/checkpoint.pth"):
                os.remove(f"./{save_dir}/checkpoint.pth")
            torch.save(model.state_dict(), f"./{save_dir}/checkpoint.pth")
            history_epochs.append(epoch)
            
        
        

            
