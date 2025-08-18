import torch
import torch.nn as nn
import re
from transformers import RobertaModel
import random
from collections import defaultdict
from torch.utils.checkpoint import checkpoint

def get_labels(triples, entity_pairs, label_to_id, num_class=97):
    labels = torch.zeros(size=(len(entity_pairs), num_class))
    for i, entity_pair in enumerate(entity_pairs):
        h, t = entity_pair
        for triple in triples:
            if triple["h"] == h and triple["t"] == t:
                r = label_to_id[triple["r"]]
                labels[i][r] = 1
    return labels

class EntityEncoder(nn.Module):
    def __init__(self, bert_model, tokenizer):
        super(EntityEncoder, self).__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.hidden_size = self.bert_model.config.hidden_size
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        # self.layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
    
    def forward(self, entities):
        device = next(self.parameters()).device
        # 获取<e>和</e>的token ID
        start_token_id = self.tokenizer.convert_tokens_to_ids('<e>')
        end_token_id = self.tokenizer.convert_tokens_to_ids('</e>')

        # 将entities列表中的句子进行编码
        inputs = self.tokenizer(entities, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 获取BERT的隐藏状态
        outputs = self.bert_model(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state  # 形状为[n, seq_len, hidden_size]

        # 初始化列表来存储结果
        entity_hidden_states = []

        # 对每个句子进行处理
        for i in range(len(entities)):
            hs = hidden_states[i]       # 形状为[seq_len, hidden_size]
            input_ids_i = input_ids[i]  # 形状为[seq_len]

            # 找到<e>和</e>的索引位置
            start_index = (input_ids_i == start_token_id).nonzero(as_tuple=True)[0][0].item()
            end_index = (input_ids_i == end_token_id).nonzero(as_tuple=True)[0][0].item()

            # 提取<e>和</e>之间（包括它们）的隐藏状态
            hs_entity = hs[start_index:end_index+1]  # 形状为[token_count, hidden_size]

            # 对隐藏状态做平均操作
            pool_hs = torch.mean(hs_entity, dim=0)  # 形状为[hidden_size]

            # 将结果添加到列表中
            entity_hidden_states.append(pool_hs)

        # 将列表转换为tensor，形状为[n, hidden_size]
        entity_hidden_states = torch.stack(entity_hidden_states, dim=0)
        cls_hidden_states = hidden_states[:, 0, :]
        
        entity_embeddings = self.linear(cls_hidden_states) + entity_hidden_states
        # entity_embeddings = self.layer_norm(entity_embeddings)
        
        return entity_embeddings
                
        

class RelationEncoder(nn.Module):
    def __init__(self, bert_model, tokenizer):
        super(RelationEncoder, self).__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.hidden_size = self.bert_model.config.hidden_size
        self.linear1 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        # self.layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
    
    def forward(self, relations):
        device = next(self.parameters()).device
        
        masked_relations = []
        for item in relations:
            item = re.sub(r'<h>.*?</h>', '[entity1]', item)
            item = re.sub(r'<t>.*?</t>', '[entity2]', item)
            masked_relations.append(item)
           

        # 将Relation列表中的句子进行编码
        inputs = self.tokenizer(masked_relations, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 获取BERT的隐藏状态
        outputs = self.bert_model(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state  # 形状为[n, seq_len, hidden_size]

        cls_hidden_states = hidden_states[:, 0, :]
        return self.linear1(cls_hidden_states)
        
# without layer_norm, with dropout, with edge_fusion
class RE(nn.Module):
    def __init__(self, bert_model, tokenizer, max_hop, num_labels):
        super(RE, self).__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.embed_size = self.bert_model.config.hidden_size
        self.classifier = nn.Linear(in_features=self.embed_size * 2, out_features=self.num_labels)
        self.head_extractor = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
        self.tail_extractor = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
        self.entity_encoder = EntityEncoder(self.bert_model, self.tokenizer)
        self.relation_encoder = RelationEncoder(self.bert_model, self.tokenizer)
        self.path_encoder = RobertaModel.from_pretrained('roberta-base')
        self.path_linear = nn.Linear(in_features=3 * self.embed_size, out_features=self.embed_size)
        # self.dropout = nn.Dropout(p=0.15)
        self.max_hop = max_hop
        
    def find_paths(self, edges, l, max_hop):
        from collections import deque, defaultdict
        
        # Build the adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
        
        res = {}
        
        for s in l:
            distances = {s: 0}
            predecessors = defaultdict(list)
            queue = deque([s])
        
            while queue:
                node = queue.popleft()
                if distances[node] >= max_hop:
                    continue
                for neighbor in adj[node]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        predecessors[neighbor].append(node)
                        queue.append(neighbor)
                    elif distances[neighbor] == distances[node] + 1:
                        predecessors[neighbor].append(node)
                        # No need to add neighbor to queue again
        
            for t in l:
                if t != s and t in distances and distances[t] <= max_hop:
                    # Collect all shortest paths from s to t
                    paths = []
                    def backtrack(current_node, path):
                        if current_node == s:
                            paths.append(list(reversed(path)))
                            return
                        for pred in predecessors[current_node]:
                            backtrack(pred, path + [(pred, current_node)])
                    backtrack(t, [])
                    res[(s, t)] = paths
        
        return res
    
    def fusion(self, hts, embeddings):
        # print("edge_fusion called")
        # 用字典存储重复的边和它们对应的嵌入
        edge_map = defaultdict(list)

        # 遍历 edges 和 embeddings，将相同的边放在一起
        for idx, ht in enumerate(hts):
            edge_map[tuple(ht)].append(idx)

        merged_hts = []
        merged_embeddings = []

        # 遍历字典中的每个边，合并重复的边
        for edge, indices in edge_map.items():
            merged_hts.append(edge)
            
            # 如果该边有多个索引，进行pooling运算
            if len(indices) > 1:
                embeddings_to_merge = embeddings[indices]
                merged_embedding = torch.mean(embeddings_to_merge, dim=0)
                # merged_embeddings = torch.mean(embeddings_to_merge, dim=0)
                merged_embeddings.append(merged_embedding)
            else:
                merged_embeddings.append(embeddings[indices[0]])

        # 转换成最终的 tensor
        merged_embeddings = torch.stack(merged_embeddings)
        
        return merged_hts, merged_embeddings

    
    def edge_fusion(self, edges, edge_embeddings):
        return self.fusion(edges, edge_embeddings)
    
    def path_fusion(self, hts, path_embeddings):
        return self.fusion(hts, path_embeddings)
    
    # def transformers_path_embedding(self, path_seqs):
    #     path_encode = self.path_encoder(inputs_embeds=path_seqs).last_hidden_state
    #     head = path_encode[:, 0, :]
    #     tail = path_encode[:, -1, :]
    #     path_embedding = torch.cat([head * tail, head + tail, torch.abs(head - tail)], dim=-1)
    #     path_embedding = self.path_linear(path_embedding)
    #     return path_embedding # [n, hidden_size]   
    
    def transformers_path_embedding(self, path_seqs, chunk_size: int = 32, threshold: int = 512):
        """
        path_seqs: Tensor [N, L, H]
        chunk_size: mini-batch size when using gradient checkpointing
        threshold: switch point for triggering chunked encoding
        """
        num_paths = path_seqs.size(0)

        # ✅ Fast path: use original method if path count is small
        if num_paths <= threshold:
            path_encode = self.path_encoder(inputs_embeds=path_seqs).last_hidden_state
            head = path_encode[:, 0, :]
            tail = path_encode[:, -1, :]
            path_embedding = torch.cat([head * tail, head + tail, torch.abs(head - tail)], dim=-1)
            path_embedding = self.path_linear(path_embedding)
            
                # ✅ 清理内存
            del path_encode, head, tail
            torch.cuda.empty_cache()
            return path_embedding

        # 🧠 Slow path: chunked + checkpointed processing
        def encode_chunk(chunk_inputs):
            return self.path_encoder(inputs_embeds=chunk_inputs).last_hidden_state

        outputs = []
        for i in range(0, num_paths, chunk_size):
            chunk = path_seqs[i:i + chunk_size]
            chunk = chunk.requires_grad_()

            path_encode = checkpoint(encode_chunk, chunk, use_reentrant=False)
            head = path_encode[:, 0, :]
            tail = path_encode[:, -1, :]
            fused = torch.cat([head * tail, head + tail, torch.abs(head - tail)], dim=-1)
            fused = self.path_linear(fused)
            outputs.append(fused)

            # ✅ 显式释放当前子 batch 的临时变量，避免显存堆积
            del chunk, path_encode, head, tail, fused
            torch.cuda.empty_cache()

        return torch.cat(outputs, dim=0)
        
    
    def forward(self, entity_description, relation_description_graph):
        device = next(self.parameters()).device
        if len(relation_description_graph) == 0:
            return {
                "pairs" : [],
                "logits" : torch.empty(size=(0, self.num_labels)).to(device)
            }
        
        
        entity_description = sorted(entity_description, key=lambda x : x["entity_id"])
        
        entity_description_sentences = [x["description"] for x in entity_description]
        relation_description_sentences = [x["description"] for x in relation_description_graph]
        edges_forward = [(x["h"], x["t"]) for x in relation_description_graph]
        edges_reversed = [(x["t"], x["h"]) for x in relation_description_graph]
        
        num_entities = len(entity_description_sentences)
        
        edges = edges_forward + edges_reversed # l = 2*relation_len
        
        entity_embeddings = self.entity_encoder(entity_description_sentences) # [entity_num, hidden_size]
        
        edge_embeddings = self.relation_encoder(relation_description_sentences)
        
        edge_embeddings = torch.cat([edge_embeddings, edge_embeddings]) # [2*relation_len, hidden_size]
        
        edges, edge_embeddings = self.edge_fusion(edges, edge_embeddings)
        
        entity_ids = list(range(num_entities))
        
        path_seq_embeddings = [[] for _ in range(self.max_hop + 1)] # [2 * n + 1, hidden_size] in n if not empty
        connected_pairs = [[] for _ in range(self.max_hop + 1)]
        
        possible_paths = self.find_paths(edges, entity_ids, max_hop=self.max_hop)
        
        for v, k in possible_paths.items():
            # num_path = len(k)
            num_hop = len(k[0]) # num of hops
            

            for path in k:
                connected_pairs[num_hop].append(v)
                entity_id_seq = [path[0][0]]
                edge_id_seq = []

                for x in path:
                    entity_id_seq.append(x[1])
                    edge_id_seq.append(edges.index(x))
                
                entity_id_seq = torch.tensor(entity_id_seq).to(device)
                edge_id_seq = torch.tensor(edge_id_seq).to(device)
                entity_embeddings_seq = entity_embeddings[entity_id_seq] #[num_hop + 1, hidden_size]
                edge_embeddings_seq = edge_embeddings[edge_id_seq] # [num_hop, hidden_size]
                
                path_seq_embedding = torch.empty(size=(2 * num_hop + 1, self.embed_size)).to(device)
                
                path_seq_embedding[0::2] = entity_embeddings_seq
                path_seq_embedding[1::2] = edge_embeddings_seq
                path_seq_embeddings[num_hop].append(path_seq_embedding)
        
        path_embeddings = []
        entity_pairs = []

        for path_seq_embedding, connected_pair in zip(path_seq_embeddings, connected_pairs): # iter num_hop times
            # i th is the i hops
            # path_seq_embedding [num_i+1_hops, 2*i + 3, hidden_size]
            # connected_pair [num_i_hops]
            if len(path_seq_embedding) == 0:
                continue
            path_embeddings_i_hop = self.transformers_path_embedding(torch.stack(path_seq_embedding)) # [num_i_hops ,hidden_size]
            pairs, path_embeddings_i_hop = self.path_fusion(connected_pair, path_embeddings_i_hop)

            path_embeddings.append(path_embeddings_i_hop)
            entity_pairs += pairs
            
            del path_embeddings_i_hop, pairs, connected_pair, path_seq_embedding
        torch.cuda.empty_cache()    
        
        path_embeddings = torch.cat(path_embeddings, dim=0) # [num_pairs, hidden_size]
        # hts, path_embeddings = self.path_fusion(hts, path_embeddings)
        # for connected_pair in connected_pairs:
        #     entity_pairs += connected_pair  

        hs = []
        ts = []

        for ht in entity_pairs:
            hs.append(ht[0])
            ts.append(ht[1])

        hs = torch.tensor(hs).to(device)
        ts = torch.tensor(ts).to(device)
        hs = entity_embeddings[hs]
        ts = entity_embeddings[ts]  

        zh = torch.tanh(self.head_extractor(hs) + path_embeddings) # [num_pairs, hidden_size]
        zt = torch.tanh(self.tail_extractor(ts) + path_embeddings)
        
        # zh = torch.nn.functional.normalize(zh, p=2, dim=1)
        # zt = torch.nn.functional.normalize(zt, p=2, dim=1)   
        # zh = self.layernorm(zh)
        # zt = self.layernorm(zt)
        
        feature = torch.cat([zh, zt], dim=1) # [num_pairs, 2 * hidden_size]
        # feature = logits
        
        logits = self.classifier(feature) # [num_pairs, 97]
        
        return {
            "pairs" : entity_pairs,
            "logits" : logits,
            "feature" : feature
        } 