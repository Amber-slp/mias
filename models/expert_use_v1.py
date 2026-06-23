import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import json
import gc 

def load_model_and_infer(base_model_path, adapter_path, user_query):
    """
    加载模型并进行推理。
    注意：为了防止显存溢出，函数结束前会尝试清理显存。
    """
    print("="*50)
    print(f"正在初始化 (启用 4-bit 量化模式)...")
    print(f"Base Model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = None
    tokenizer = None
    decoded_output = "推理失败"

    try:
        # 1. 加载 Tokenizer
        print("正在加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            trust_remote_code=True
        )

        # 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 2. 加载原始底座模型
        print("正在加载底座模型 (4-bit 量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto" 
        )

        # 3. 加载 LoRA 适配器并合并
        print("正在挂载 LoRA 微调权重...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # 切换到评估模式
        model.eval()
        
        print("模型加载成功！开始生成答案...")
        print("-" * 30)

        # 4. 构建输入
        # 根据模型不同，这里可能需要调整 Prompt Template
        messages = [
            {"role": "user", "content": user_query}
        ]
        
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        except Exception:
            # 如果没有 chat template，回退到直接输入
            input_ids = tokenizer(user_query, return_tensors="pt").input_ids.to(model.device)

        # 5. 生成答案
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1 
            )

        # 6. 解码输出
        response = outputs[0][input_ids.shape[-1]:]
        decoded_output = tokenizer.decode(response, skip_special_tokens=True)
        print("生成完成。")

    except Exception as e:
        import traceback
        traceback.print_exc()
        decoded_output = f"发生错误: {str(e)}"
    
    finally:
        # 7. 显存清理 (至关重要)
        # 因为是在循环中反复加载模型，必须手动释放资源
        print("正在清理显存资源...")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("资源已释放。")

    return decoded_output

def load_model_configs(config_file="model_dir.json"):
    """加载模型目录配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            model_configs = json.load(f)
        return model_configs
    except Exception as e:
        print(f"加载模型配置文件失败: {e}")
        return []

def load_questions(question_file="question.json"):
    """加载问题文件"""
    try:
        with open(question_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions
    except Exception as e:
        print(f"加载问题文件失败: {e}")
        return []

def find_model_config(model_configs, expert_name):
    """根据专家名称查找对应的模型配置"""
    for config in model_configs:
        # 忽略大小写比较，增加鲁棒性
        if config.get("expert", "").lower() == expert_name.lower():
            return config
    return None

def process_questions():
    """主处理函数"""
    
    # 1. 加载配置
    model_configs = load_model_configs()
    if not model_configs:
        print("错误：未找到 model_dir.json 或文件为空。")
        return
    
    questions = load_questions("question.json")
    if not questions:
        print("错误：未找到 question.json 或文件为空。")
        return
    
    print(f"成功加载 {len(model_configs)} 个模型配置")
    print(f"成功加载 {len(questions)} 个问题")
    
    results = []
    
    # 2. 遍历处理每个问题
    for i, question_data in enumerate(questions, 1):
        # --- 适配点 1: 字段名称映射 ---
        # 上一步输出的 key 是 "original_question" 和 "analysis_result"
        question_text = question_data.get("original_question", "")
        experts_analysis = question_data.get("analysis_result", [])
        q_id = question_data.get("id", i)
        
        print(f"\n[{i}/{len(questions)}] 处理问题 (ID: {q_id}): {question_text[:30]}...")
        
        expert_answers = []
        
        # 遍历该问题涉及的所有专家领域
        for expert_info in experts_analysis:
            expert_name = expert_info.get("expert", "")
            weight = expert_info.get("weight", 0)
            reason = expert_info.get("reason", "")
            
            print(f"  -> 调用专家模型: {expert_name} (权重: {weight})")
            
            # 查找配置
            model_config = find_model_config(model_configs, expert_name)
            
            if not model_config:
                print(f"     [警告] 未在 model_dir.json 中找到专家 '{expert_name}' 的配置，跳过。")
                expert_answer = "Error: No model configuration found for this domain."
            else:
                base_path = model_config.get("BASE_MODEL_PATH")
                adapter_path = model_config.get("ADAPTER_PATH")
                
                # 执行推理
                expert_answer = load_model_and_infer(
                    base_path, 
                    adapter_path, 
                    question_text # 将原始问题传给专家模型
                )
            
            # 构建结果条目
            expert_answers.append({
                "expert": expert_name,
                "weight": weight,
                "reason": reason,
                "answer": expert_answer
            })
        
        # 汇总该问题的结果
        results.append({
            "id": q_id,
            "question": question_text,
            "experts_output": expert_answers
        })
    
    # 3. 保存最终结果
    try:
        with open("answer.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n全部完成！答案已保存至 answer.json")
    except Exception as e:
        print(f"保存结果文件失败: {e}")

if __name__ == "__main__":
    process_questions()
