import json
import os

def merge_answer_files(file1_path, file2_path, output_path):
    """
    读取两个JSON文件，根据问题字段匹配答案，并生成新的JSON文件。
    
    Args:
        file1_path (str): final_dense_answer.json 的路径
        file2_path (str): answer_mix.json 的路径
        output_path (str): 输出文件 answer_3.json 的路径
    """
    
    # 1. 读取 final_dense_answer.json
    print(f"正在读取 {file1_path} ...")
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
            # 确保数据是列表格式，如果不是列表（例如是单个对象），则放入列表中
            if isinstance(data1, dict):
                data1 = [data1]
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file1_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: {file1_path} 不是有效的 JSON 文件")
        return

    # 2. 读取 answer_mix.json
    print(f"正在读取 {file2_path} ...")
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
            if isinstance(data2, dict):
                data2 = [data2]
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file2_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: {file2_path} 不是有效的 JSON 文件")
        return

    # 3. 预处理 data2 以便快速查找
    # 创建一个字典，key是问题，value是答案。这样查找的时间复杂度是 O(1)
    print("正在建立索引以进行匹配...")
    answer_mix_map = {}
    for item in data2:
        # 假设 json 中的字段名确实是 "question" 和 "answer"
        # 使用 .get() 防止字段不存在报错，strip() 去除可能存在的首尾空格以提高匹配率
        q = item.get("question", "").strip()
        a = item.get("answer", "")
        if q:
            answer_mix_map[q] = a

    # 4. 拼接数据
    print("正在合并数据...")
    merged_results = []
    
    matched_count = 0
    missing_count = 0

    for item in data1:
        question = item.get("question", "").strip()
        answer1 = item.get("synthesized_answer", "")
        
        # 在 answer_mix_map 中查找对应的答案2
        answer2 = answer_mix_map.get(question)
        
        if answer2 is not None:
            matched_count += 1
        else:
            missing_count += 1
            answer2 = "未在answer_mix.json中找到匹配的问题" # 或者设置为 None/空字符串

        # 构建新的数据对象
        new_entry = {
            "question": question,
            "answer1": answer1,
            "answer2": answer2
        }
        merged_results.append(new_entry)

    # 5. 写入 answer_3.json
    print(f"正在写入 {output_path} ...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文能正常显示，indent=4 保证格式美观
            json.dump(merged_results, f, ensure_ascii=False, indent=4)
        print("写入完成！")
        print(f"统计: 共处理 {len(merged_results)} 条数据。")
        print(f"      成功匹配: {matched_count} 条")
        print(f"      未匹配到: {missing_count} 条")
        
    except IOError as e:
        print(f"写入文件时发生错误: {e}")

if __name__ == "__main__":
    # 定义文件名
    file_dense = "final_dense_answer.json"
    file_mix = "answer_mix.json"
    file_output = "answer_3.json"

    # 为了测试方便，如果文件不存在，这里可以生成一些假数据（可选）
    # 如果你已有真实文件，请忽略以下生成假数据的代码块
    if not os.path.exists(file_dense) or not os.path.exists(file_mix):
        print("注意：未检测到输入文件，正在生成示例文件用于演示...")
        sample_dense = [
            {"question": "天空为什么是蓝色的？", "synthesized_answer": "因为瑞利散射。"},
            {"question": "Python是什么？", "synthesized_answer": "一种编程语言。"}
        ]
        sample_mix = [
            {"question": "天空为什么是蓝色的？", "answer": "因为大气层散射太阳光。"},
            {"question": "地球是圆的吗？", "answer": "是的。"}
        ]
        with open(file_dense, 'w', encoding='utf-8') as f:
            json.dump(sample_dense, f, ensure_ascii=False)
        with open(file_mix, 'w', encoding='utf-8') as f:
            json.dump(sample_mix, f, ensure_ascii=False)
        print("示例文件生成完毕。\n")

    # 执行主函数
    merge_answer_files(file_dense, file_mix, file_output)
