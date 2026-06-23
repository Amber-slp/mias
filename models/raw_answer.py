import json
import requests
import sys
import re
import time
import threading  


API_KEY = ""
API_URL = "https://api.deepseek.com/v1/chat/completions"
INPUT_FILE = "seed.txt"
OUTPUT_FILE = "answer_raw_model.json"
BATCH_SIZE = 3  


results = []
results_lock = threading.Lock()

def get_model_answer(question, index):


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": question}
        ],
        "stream": False
    }

    result_entry = {
        "id": index,
        "question": question,
        "answer": None,
        "error": None
    }

    try:

        response = requests.post(API_URL, headers=headers, json=payload, timeout=500)
        
        if response.status_code == 200:
            data = response.json()
            try:
                answer = data['choices'][0]['message']['content']
                result_entry["answer"] = answer

            except (KeyError, IndexError) as e:
                result_entry["error"] = f": {str(e)}"

        else:
            result_entry["error"] = f" {response.status_code}: {response.text}"
            print(f" {index}: HTTP {response.status_code}")
            
    except Exception as e:
        result_entry["error"] = f": {str(e)}"
        print(f" {index}: {str(e)}")


    with results_lock:
        results.append(result_entry)

def main():

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:

            questions = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:

        sys.exit(1)

    total = len(questions)

    
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch_questions = questions[i : i + BATCH_SIZE]
        threads = []

        for j, question in enumerate(batch_questions):
            global_index = i + j + 1
            t = threading.Thread(target=get_model_answer, args=(question, global_index))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    results.sort(key=lambda x: x['id'])

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    except IOError as e:
        print(f" {e}")

if __name__ == "__main__":
    main()
