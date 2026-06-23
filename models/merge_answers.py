import json
import os

def load_json(file_path):
    """
    辅助函数：加载JSON文件，如果文件不存在或格式错误则返回空列表/字典并打印警告。
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在。")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"警告: 文件 {file_path} 的根元素不是列表。")
                return []
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的 JSON 格式。")
        return []

def create_lookup_dict(data_list, key_field='question'):
    """
    辅助函数：将列表转换为以 question 为键的字典，方便快速查找。
    """
    lookup = {}
    for item in data_list:
        if key_field in item:
            # 移除首尾空格以增加匹配成功率
            q_key = item[key_field].strip()
            lookup[q_key] = item
    return lookup

def main():
    # 1. 定义文件名
    file_answer_3 = 'answer_3.json'
    file_final = 'final.json'
    file_raw_model = 'answer_raw_model.json'
    output_file = 'all_answer.json'

    print("开始读取文件...")

    # 2. 读取所有源文件
    data_answer_3 = load_json(file_answer_3)
    data_final = load_json(file_final)
    data_raw_model = load_json(file_raw_model)

    if not data_answer_3:
        print("错误: 主数据源 (answer_3.json) 为空或读取失败，程序终止。")
        return

    # 3. 创建查找字典 (Lookup Dictionaries)
    print("正在建立索引...")
    dict_final = create_lookup_dict(data_final, 'question')
    dict_raw_model = create_lookup_dict(data_raw_model, 'question')

    merged_results = []

    print("正在合并数据并生成新ID...")
    
    # 4. 遍历主数据源 (answer_3.json) 进行匹配
    # 使用 enumerate(..., start=1) 来生成从1开始的顺序ID
    for index, item_3 in enumerate(data_answer_3, start=1):
        
        # 获取问题文本用于匹配
        question_text = item_3.get('question', '').strip()
        
        # 如果问题为空，一般跳过，或者你可以选择保留但标记为空
        if not question_text:
            continue

        # 初始化合并后的对象
        merged_item = {
            "id": index,  # 这里强制使用生成的顺序 ID (1, 2, 3...)
            "question": question_text,
            "answer1": "",
            "answer2": "",
            "answer3": "",
            "answer4": ""
        }

        # --- 填充 Answer 2 和 Answer 3 (来自 answer_3.json) ---
        # 规则：answer_3中的 "answer1" -> 新文件的 answer2
        merged_item["answer2"] = item_3.get('answer1', "")
        # 规则：answer_3中的 "answer2" -> 新文件的 answer3
        merged_item["answer3"] = item_3.get('answer2', "")

        # --- 填充 Answer 1 (来自 final.json) ---
        # 规则：final.json中的 "final_answer" -> 新文件的 answer1
        if question_text in dict_final:
            match_final = dict_final[question_text]
            merged_item["answer1"] = match_final.get('final_answer', "")

        # --- 填充 Answer 4 (来自 answer_raw_model.json) ---
        # 规则：answer_raw_model.json中的 "answer" -> 新文件的 answer4
        if question_text in dict_raw_model:
            match_raw = dict_raw_model[question_text]
            merged_item["answer4"] = match_raw.get('answer', "")

        merged_results.append(merged_item)

    # 5. 写入结果文件
    print(f"合并完成，共处理 {len(merged_results)} 条数据。")
    print(f"正在写入 {output_file} ...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文正常显示
            # indent=4 保证输出格式美观
            json.dump(merged_results, f, ensure_ascii=False, indent=4)
        print("写入成功！")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

if __name__ == "__main__":
    main()
#1是mias之后的答案，2是专家整合后的，3是mix的，4是原生模型的。
