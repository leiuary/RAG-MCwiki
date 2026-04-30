import os
import json
import re

INPUT_DIR = 'data/output'
OUTPUT_DIR = 'data/structured_output'

def deduplicate_line(line):
    """去除行内重复的单词并清理占位符"""
    # 彻底移除所有 [[|]] 及其变体
    line = re.sub(r'\[\[\|\]\]', '', line).strip()
    
    if not line:
        return ""
        
    words = line.split()
    if not words:
        return ""
    new_words = []
    for w in words:
        if not new_words or w != new_words[-1]:
            new_words.append(w)
    return " ".join(new_words)

def clean_and_structure(raw_text, title):
    lines = raw_text.split('\n')
    
    # 初步清洗
    valid_lines = []
    for line in lines:
        line = line.strip()
        if not line or line in ("导航", "资源", "可再生产品", "可再生获取方法"):
            continue
        # 过滤纯占位符、破折号和无意义符号
        if not re.sub(r'(\[\[\|\]\])+|—', '', line).strip():
            continue
        # 过滤消歧义常用语
        if "这个消歧义页面列出了" in line or "如果您是通过某条目" in line:
            continue
            
        line = deduplicate_line(line)
        if line:
            valid_lines.append(line)

    structured_data = {}
    current_section = "通用"
    current_entity = None
    
    # 常见的描述起始词
    method_starters = ("由", "从", "在", "通过", "进入", "和", "与", "对", "使用", "让", "当", "淇", "鍙犲姞") # 包含部分乱码对应的起始词
    
    for line in valid_lines:
        # 1. 识别大节标题
        header_match = re.match(r"^##\s*(.*?)\s*##$", line)
        if header_match:
            current_section = header_match.group(1)
            structured_data[current_section] = {}
            current_entity = None
            continue

        # 2. 识别实体与描述
        # 判断标准：如果一行很短，且不以描述词开头，且不包含句号，认为是实体名
        is_description = (
            line.startswith(method_starters) or 
            line.endswith(('。', '！', '？', ':', '：')) or 
            len(line) > 30 or
            any(char.isdigit() for char in line)
        )
        
        if not is_description:
            if current_section not in structured_data:
                structured_data[current_section] = {}
            
            # 优化：如果当前实体已经有内容了，新遇到的短行大概率是新实体
            if current_entity is None or len(structured_data[current_section].get(current_entity, [])) > 0:
                current_entity = line
                if current_entity not in structured_data[current_section]:
                    structured_data[current_section][current_entity] = []
            else:
                # 连续的短行，认为是同一实体的补充说明（如图标后的名称）
                structured_data[current_section][current_entity].append(line)
        else:
            # 这是一条具体的描述
            if current_entity:
                if current_section not in structured_data:
                    structured_data[current_section] = {}
                if current_entity not in structured_data[current_section]:
                    structured_data[current_section][current_entity] = []
                structured_data[current_section][current_entity].append(line)
            else:
                # 兜底
                if "未分类信息" not in structured_data:
                    structured_data["未分类信息"] = {}
                if "备注" not in structured_data["未分类信息"]:
                    structured_data["未分类信息"]["备注"] = []
                structured_data["未分类信息"]["备注"].append(line)

    # 最终清理：移除空条目
    final_data = {}
    for sec, entities in structured_data.items():
        clean_entities = {k: v for k, v in entities.items() if v}
        if clean_entities:
            final_data[sec] = clean_entities

    return final_data

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.json') or filename.startswith('global_index'):
            continue
            
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(input_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            raw_text = data.get("text", "")
            title = data.get("title", "")
            
            data["structured_content"] = clean_and_structure(raw_text, title)
            
            if "text" in data:
                del data["text"]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ 成功结构化: {filename}")
            
        except Exception as e:
            print(f"❌ 处理 {filename} 时发生错误: {e}")

if __name__ == "__main__":
    main()
