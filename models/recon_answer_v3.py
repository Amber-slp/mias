import json
import requests
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- CONFIGURATION ---
API_KEY = ""  # Replace with your actual key
API_URL = "https://api.deepseek.com/v1/chat/completions"
INPUT_FILE = "answer.json"
OUTPUT_FILE = "final_dense_answer.json"
MAX_WORKERS = 3  # 并发线程数

# 用于在多线程中统计进度的锁和计数器
progress_lock = threading.Lock()
completed_count = 0
total_count = 0

def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load input file: {e}")
        sys.exit(1)

def clean_json_string(s):
    """
    Cleans the raw string from the LLM to ensure it can be parsed as JSON.
    Removes Markdown code blocks and whitespace.
    """
    s = re.sub(r'^```json\s*', '', s, flags=re.MULTILINE)
    s = re.sub(r'^```\s*', '', s, flags=re.MULTILINE)
    s = re.sub(r'\s*```$', '', s, flags=re.MULTILINE)
    return s.strip()

def process_question(item):
    """
    Worker function to process a single item.
    Returns the final formatted dictionary or None if failed.
    """
    global completed_count
    
    question_text = item.get('question', 'Unknown Question')
    experts = item.get('experts', [])
    
    # Construct the context for the model
    experts_context = ""
    for idx, exp in enumerate(experts):
        role = exp.get('expert', 'General Expert')
        weight = exp.get('weight', 0.5)
        content = exp.get('answer', '')
        
        experts_context += f"""
        [Source #{idx+1}]
        - Role/Specialty: {role}
        - Reliability Weight: {weight}
        - Content: {content}
        """

    # --- SYSTEM PROMPT ---
    system_prompt = """
    You are an advanced Scientific Knowledge Synthesis Engine. Your task is to generate a single, high-information-density answer by synthesizing inputs from multiple experts.

    **STRICT OUTPUT RULES:**
    1. Output MUST be valid JSON only. No conversational text.
    2. Output language MUST be pure English.

    **SYNTHESIS STRATEGY:**
    1. **Domain Specialization**: Pay attention to the 'Role/Specialty' of the source.
    2. **Weighted Integration**: For general facts, prioritize high-weight sources.
    3. **Numerical Sanity Check**: If experts provide vastly different numbers, use internal knowledge to judge.
    4. **Correction**: Fix obvious OCR errors.

    **JSON STRUCTURE:**
    {
      "synthesized_answer": "The final, dense academic text...",
      "synthesis_report": {
        "domain_utilization": "Explain which expert was prioritized...",
        "conflict_resolution": "Explain how conflicts were resolved...",
        "corrections": "List specific fixes made..."
      }
    }
    """

    user_prompt = f"""
    **Target Question**: "{question_text}"
    
    **Expert Inputs**:
    {experts_context}
    
    Generate the JSON response based on the strategy defined.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2500
    }

    result_data = None

    try:
        # Retry logic for stability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                
                raw_content = response.json()['choices'][0]['message']['content']
                cleaned_json = clean_json_string(raw_content)
                parsed_json = json.loads(cleaned_json)
                
                # Construct the final record immediately
                result_data = {
                    "question": question_text,
                    "synthesized_answer": parsed_json.get("synthesized_answer"),
                    "synthesis_report": parsed_json.get("synthesis_report")
                }
                break # Success, exit retry loop
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2) # Wait before retry
                    continue
                else:
                    print(f"\n[Error] Failed processing: {question_text[:30]}... | Error: {e}")

    except Exception as e:
        print(f"\n[Critical Error] Unexpected error: {e}")

    # Update progress bar
    with progress_lock:
        completed_count += 1
        print(f"\rProgress: [{completed_count}/{total_count}]", end="", flush=True)

    return result_data

def main():
    global total_count
    print("Loading data...")
    data = load_data(INPUT_FILE)
    total_count = len(data)
    final_output = []

    print(f"Starting processing with {MAX_WORKERS} concurrent workers for {total_count} items...")
    start_time = time.time()

    # Using ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(process_question, item): item for item in data}
        
        # Process as they complete
        for future in as_completed(future_to_item):
            result = future.result()
            if result:
                final_output.append(result)

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n\nProcessing complete in {duration:.2f} seconds.")
    print(f"Successful: {len(final_output)} / {total_count}")

    # Save to file
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        print(f"Success! Output saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
