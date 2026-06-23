import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import json
import os


BASE_MODEL_PATH = 
ADAPTER_PATH = 

INPUT_FILE = "seed.txt"
OUTPUT_FILE = "answer_mix.json"


def initialize_model(base_path, adapter_path):

    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print(f" {base_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_path, 
            trust_remote_code=True
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        print("正在加载底座模型 (4-bit 量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto" 
        )

        print(f"正在挂载 LoRA 微调权重: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        
        print("模型加载完成！")
        print("-" * 30)
        
        return model, tokenizer

    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_answer(model, tokenizer, user_query):
    """
    使用已加载的模型进行单次推理
    """
    try:
        # 构建输入
        messages = [
            {"role": "user", "content": user_query}
        ]
        
        # 尝试应用聊天模板
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        except Exception:
            # 如果模板应用失败或模型没有模板，直接编码文本
            input_ids = tokenizer(user_query, return_tensors="pt").input_ids.to(model.device)

        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码输出 (只取新生成的token)
        response = outputs[0][input_ids.shape[-1]:]
        decoded_output = tokenizer.decode(response, skip_special_tokens=True)

        return decoded_output

    except Exception as e:
        return f"生成出错: {str(e)}"

def load_questions_from_txt(file_path):
    """从 txt 文件加载问题，每行一个"""
    if not os.path.exists(file_path):
        print(f"错误: 找不到输入文件 {file_path}")
        return []
    
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: # 跳过空行
                questions.append(line)
    return questions

def process_questions():
    """主处理流程"""
    
    # 1. 读取问题
    questions = load_questions_from_txt(INPUT_FILE)
    if not questions:
        print("没有读取到有效的问题，程序退出。")
        return
    
    print(f"从 {INPUT_FILE} 成功读取 {len(questions)} 个问题。")

    # 2. 初始化模型 (只做一次)
    model, tokenizer = initialize_model(BASE_MODEL_PATH, ADAPTER_PATH)
    
    # 3. 批量推理
    results = []
    total = len(questions)
    
    for i, question_text in enumerate(questions, 1):
        print(f"[{i}/{total}] 正在处理: {question_text[:50]}..." if len(question_text) > 50 else f"[{i}/{total}] 正在处理: {question_text}")
        
        answer = generate_answer(model, tokenizer, question_text)
        
        # 构建结果对象
        result_item = {
            "id": i,
            "question": question_text,
            "answer": answer
        }
        results.append(result_item)
        
        # 可选：实时打印一部分答案预览
        # print(f"   -> 回答预览: {answer[:50]}...")

    # 4. 保存结果
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n所有处理完成！答案已保存到 {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存结果文件失败: {e}")

if __name__ == "__main__":
    process_questions()
