import json
import requests
import os
import time

# =========================
# API Configuration
# =========================
API_KEY = ""
API_URL = "https://api.deepseek.com/v1/chat/completions"

# =========================
# GLOBAL SETTINGS
# =========================
TIMEOUT = 60          # 防卡死
MAX_RETRY = 5         # 最大重试
SLEEP_BETWEEN = 1     # 限速
MAX_CHARS = 2000      # 防 token 爆炸

session = requests.Session()

# =========================
# CORE FUNCTION
# =========================
def generate_refined_answer(question, answer1, answer2):
    system_prompt = """
    You are a senior expert in chemistry and materials science.
    Synthesize a final answer based on Answer1 (primary) and Answer2 (secondary).
    Follow strict scientific accuracy. Output only the final answer in academic English.
    """

    # 🔥 限制长度（关键！）
    answer1 = answer1[:MAX_CHARS]
    answer2 = answer2[:MAX_CHARS]

    user_content = f"""
    Question: {question}
    Answer1: {answer1}
    Answer2: {answer2}
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    for attempt in range(MAX_RETRY):
        try:
            print(f"    → Attempt {attempt+1}")

            response = session.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT
            )

            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            print("    → Success")
            return content

        except Exception as e:
            wait_time = 2 ** attempt
            print(f"    → Error: {e}")
            print(f"    → Retry in {wait_time}s...")

            time.sleep(wait_time)

    print("    → Failed after max retries")
    return None


# =========================
# MAIN PIPELINE
# =========================
def main():
    input_file = "answer_3.json"
    output_file = "final.json"

    # 1. Load data
    if not os.path.exists(input_file):
        print("Input file not found!")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    print(f"Loaded {len(data)} items\n")

    results = []

    # 2. Process
    for i, item in enumerate(data):
        print(f"\n=== Processing {i+1}/{len(data)} ===")

        q = item.get("question", "")
        a1 = item.get("answer1", "")
        a2 = item.get("answer2", "")

        if not q or not a1:
            print("  → Skipped (missing data)")
            continue

        start_time = time.time()

        result = generate_refined_answer(q, a1, a2)

        elapsed = time.time() - start_time
        print(f"  → Time used: {elapsed:.2f}s")

        if result:
            results.append({
                "question": q,
                "final_answer": result
            })
        else:
            print("  → Failed, skip")

        # 🔥 限速（防止被封）
        time.sleep(SLEEP_BETWEEN)

    # 3. Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n========================")
    print(f"Done! {len(results)} results saved to {output_file}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()
