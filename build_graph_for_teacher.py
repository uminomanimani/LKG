import json
import re
import argparse

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data

def wrap_words_in_s(s, words):
    # 查找所有 <h>...</h> 标签的范围
    h_ranges = []
    for h_match in re.finditer(r'<h>.*?</h>', s, flags=re.DOTALL):
        h_ranges.append((h_match.start(), h_match.end()))

    # 查找所有 <t> ... </t> 标签的范围
    t_ranges = []
    for t_match in re.finditer(r'<t> .*? </t>', s, flags=re.DOTALL):
        t_ranges.append((t_match.start(), t_match.end()))

    # 检查位置是否与任何范围重叠的函数
    def is_within_ranges(pos_start, pos_end, ranges):
        for (start, end) in ranges:
            if pos_start < end and pos_end > start:
                return True
        return False

    # 按单词长度降序排序
    words = sorted(words, key=lambda x: -len(x))

    # 转义单词列表中的特殊字符
    escaped_words = [re.escape(word) for word in words]

    # 定义边界单词字符，不包括句点
    word_boundary_chars = r'[A-Za-z0-9_-]'
    # 定义内部单词字符，包括句点
    word_chars = r'[A-Za-z0-9_.-]'

    # 构建带有自定义边界的正则表达式模式
    patterns = [r'(?<!{0}){1}(?!{0})'.format(word_boundary_chars, word) for word in escaped_words]
    word_pattern = r'(?:' + '|'.join(patterns) + r')'

    # 记录匹配项及其范围
    matches = []
    match_ranges = []

    # 查找匹配
    for match in re.finditer(word_pattern, s):
        match_start = match.start()
        match_end = match.end()

        # 检查匹配是否在 <h>...</h> 标签内
        if is_within_ranges(match_start, match_end, h_ranges):
            continue  # 如果在 <h>...</h> 内，跳过

        # 检查匹配是否在 <t> ... </t> 标签内
        if is_within_ranges(match_start, match_end, t_ranges):
            continue  # 如果已经被包裹，跳过

        # 检查匹配是否与已匹配的其他单词重叠
        if is_within_ranges(match_start, match_end, match_ranges):
            continue  # 如果重叠，跳过

        # 添加匹配项
        matches.append((match_start, match_end))
        # 添加到匹配范围列表
        match_ranges.append((match_start, match_end))

    # 逆序排序匹配项以避免索引偏移
    matches.sort(reverse=True, key=lambda x: x[0])

    # 对字符串应用插入
    s_list = list(s)
    for match_start, match_end in matches:
        # 在 match_end 处插入 ' </t>'
        s_list[match_end:match_end] = list(' </t>')
        # 在 match_start 处插入 '<t> '
        s_list[match_start:match_start] = list('<t> ')

    # 重新组合字符串
    result = ''.join(s_list)
    return result

def filter_tag(input_string):
    count = [0]  # Use a list to make it mutable in the nested function

    def replace_function(match):
        if count[0] == 0:
            count[0] += 1
            return match.group()
        else:
            content = match.group()[3:-4].strip()
            return content

    result = re.sub(r'<e>.*?</e>', replace_function, input_string)
    return result

def clause_clean(description):
    clause_starters = ['when', 'where', 'if', 'because', 'since', 'although', 'though', 'while', 'after', 'before', 'unless', 'who', 'in which', 'on which', 'at which', 'to which', 'for which', 'with which', 'by which', 'about which', 'from which', 'under which', 'into which', 'through which', 'which']
    t_index = description.find("</t>")
    
    # 如果找到 </t>
    if t_index != -1:
        # 截取从 </t> 之后的部分
        post_t_text = description[t_index + len("</t>"):]

        # 遍历所有从句引导词，找到最先出现的一个
        for starter in clause_starters:
            starter_index = post_t_text.find(starter)
            
            # 如果找到引导词，则截取并删除从引导词开始的部分
            if starter_index != -1:
                description = description[:t_index + len("</t>") + starter_index]
                description = description + "."
                break
        
        
    t_end_pos = description.find("</t>")
    comma_pos = description.find(", ", t_end_pos)
    if comma_pos != -1:
        description = description[:comma_pos] + "."
    return description

def sub_word(a, b):
    for string_a in a:
        for string_b in b:
            # 使用正则表达式匹配单词边界，确保是单独的单词
            pattern = r'\b' + re.escape(string_a) + r'\b'
            if re.search(pattern, string_b, re.IGNORECASE) and string_a != string_b:
                return True
    return False

def filter_mentions(s, mentions):
    import re
    to_remove = set()
    for i in mentions:
        for j in mentions:
            if i.lower() == j.lower():
                continue
            # Check if both i and j can be matched in s as whole words, ignoring case
            pattern_i = r'\b' + re.escape(i) + r'\b'
            pattern_j = r'\b' + re.escape(j) + r'\b'
            if (re.search(pattern_i, s, flags=re.IGNORECASE) and
                re.search(pattern_j, s, flags=re.IGNORECASE)):
                # Check if i can be matched inside j as a whole word
                if re.search(pattern_i, j, flags=re.IGNORECASE):
                    to_remove.add(i.lower())
    # Build the filtered list, removing mentions in a case-insensitive way
    mentions_filtered = [m for m in mentions if m.lower() not in to_remove]
    return mentions_filtered

def build_graph(entities, all_mentions):
    for m in all_mentions:
        m = sorted(m, key=len, reverse=True)

    edges = []
    # labels = original_data[text_id]["labels"]
    entities_desc = []

    for entity in entities:
        entity_id = entity["entity_id"]
        description = entity["description"]
        
        description = filter_tag(description)
        
        entities_desc.append({"entity_id" : entity_id, "description" : description})
        
        description = description.replace("<e>", "<h>")
        description = description.replace("</e>", "</h>")

        for i, v in enumerate(all_mentions):
            if i == entity_id:
                continue
            marked_description = description
            mentions = v
            
            flag = False
            try:
                for k in range(len(all_mentions)): # 检查当前实体是否为其他实体的子短语
                    if k != i:
                        m_k = all_mentions[k]
                    
                        if sub_word(mentions, m_k):
                            for l in m_k:
                                if l in marked_description:
                                    raise StopIteration()
            except StopIteration:
                flag = True
                
            if flag:
                continue
            mentions = filter_mentions(marked_description, mentions)
            marked_description = wrap_words_in_s(marked_description, mentions)
            if marked_description != description:
                marked_description = clause_clean(marked_description)
                edges.append({"h" : entity_id, "t" : i, "description" : marked_description})
    return edges, entities_desc

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--teacher_model", type=str, default="deepseek-chat", help="The teacher model which will be requested.")
    argparse.add_argument("--dataset", type=str, default="docred", help="The dataset to use.")
    argparse.add_argument("--dataset_type", type=str, default="train", help="The dataset type to use, train, dev or test.")
    args = argparse.parse_args()
    teacher_model = args.teacher_model
    dataset = args.dataset
    dataset_type = args.dataset_type

    dataset_name = json.load(open("./dataset_name.json", "r"))
    data_file = dataset_name[dataset][dataset_type]

    request_data = read_json(f"./saves/{teacher_model}/{dataset}/teacher_output/entity_description_{dataset_type}.json")
    original_data = read_json(f"./dataset/{dataset}/data/{data_file}")
    output_path = f"./saves/{teacher_model}/{dataset}/teacher_output/{dataset_type}_graph.json"
    output = []
    for count, item in enumerate(request_data):
        text_id = item["text_id"]
        entities = item["entity_descriptions"]

        vertexSet = original_data[text_id]["vertexSet"]


        if len(entities) != len(vertexSet):
            print(text_id)
            continue

        all_mentions = [[] for _ in range(len(vertexSet))]
        
        for i, v in enumerate(vertexSet):
            for m in v:
                if m["name"] not in all_mentions[i]:
                    all_mentions[i].append(m["name"])
        
        edges, entities_desc = build_graph(entities, all_mentions)


        output.append({"text_id" : text_id, "edges" : edges, "entities" : entities_desc})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    



            



    
