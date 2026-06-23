import json
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
API_KEY = "" 
API_URL = "https://api.deepseek.com/v1/chat/completions"

INPUT_FILE = "seed.txt"
OUTPUT_FILE = "question.json"
MAX_WORKERS = 1  # Parallel concurrency limit

# ==================================================
# Deep Optimization Prompt: 4-Tier Quantifiable Scoring
# ==================================================
SYSTEM_PROMPT = """
You are a top-tier Scientific Task Dispatcher for MOFs. 
Your goal is to map questions to sub-domains with quantifiable weights based on a 4-tier scoring system.

# Domain Definitions (Strictly map to these keys)
1. **Synthesis**: Preparation methods, reaction conditions, precursors, crystal growth, activation, defects.
2. **Structure**: Crystal structure, topology, pore distribution, surface area, lattice parameters, characterization (XRD, TEM, BET).
3. **Properties**: Physical/chemical properties (band gap, magnetism, mechanical strength, stability). *Note: If asking "how to use", map to Applications.*
4. **Applications**: Specific usage (gas storage, catalysis, drug delivery, sensing, environment).
5. **Theoretical Calculation**: DFT, GCMC, MD, high-throughput screening, machine learning, adsorption energy.
6. **Mechanism**: Charge transport, reaction mechanism, kinetics, electronic structure.

# Quantification Standard (Score 0, 1, 5, 10)
Assign a "Relevance Score" to RELEVANT domains using this rubric:
- **10 (Primary Intent)**: The core goal. User explicitly asks "How to X", "Calculate Y", or focuses deeply on this.
- **5 (Secondary Intent/Constraint)**: Necessary context or constraint. (e.g., "Conductive MOF" implies Mechanism context, but if asking for synthesis, Mechanism is 5).
- **1 (Mentioned/Minor)**: The keyword appears or is weakly related, but not the focus.
- **0 (Irrelevant)**: Not related.

# Calculation Logic
1. **Score**: Assign scores (10, 5, 1, 0) to all relevant domains.
2. **Filter**: Keep ALL domains with **Score >= 1**.
3. **Normalize**: Calculate weights so they sum to 1.0 based on the selected scores.
   - Formula: Weight_i = Score_i / Sum(All_Selected_Scores)
   - Example: Synthesis(10) + Structure(1) -> Total 11.
   - Synthesis Weight = 10/11 ≈ 0.91
   - Structure Weight = 1/11 ≈ 0.09

# Output Format
- Output **ONLY** a valid JSON object.
- JSON structure: 
{
  "analysis": [
    {
        "expert": "Domain Name", 
        "score": 10,
        "weight": 0.91, 
        "reason": "Explicitly asks for preparation (Score 10)."
    },
    {
        "expert": "Another Domain", 
        "score": 1,
        "weight": 0.09, 
        "reason": "Mentioned as a property check (Score 1)."
    }
  ]
}
"""

def call_deepseek_api(question, retry_count=0):
    """
    Sends a request to the API. Includes basic retry logic for rate limits.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat", 
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "temperature": 0.1, 
        "stream": False,
        "response_format": {"type": "json_object"} # Enforce JSON mode if supported, otherwise prompt handles it
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
        
        # Handle Rate Limiting (429)
        if response.status_code == 429:
            if retry_count < 3:
                wait_time = 5 * (retry_count + 1)
                print(f"[Rate Limit] 429 encountered for '{question[:20]}...'. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return call_deepseek_api(question, retry_count + 1)
            else:
                print(f"[Failed] Rate limit exceeded max retries for: {question[:20]}...")
                return None

        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Clean the response string to ensure valid JSON
        content = content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
            
        return json.loads(content)
    except Exception as e:
        print(f"[Error] Processing '{question[:20]}...': {e}")
        return None

def process_single_question(question):
    """
    Wrapper function to process a single question.
    """
    data = call_deepseek_api(question)
    if data and 'analysis' in data:
        return {"question": question, "analysis_result": data['analysis']}
    return None

def main():

    # 2. Read questions
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions. Starting parallel processing with {MAX_WORKERS} workers...")
    
    # Store results in a dictionary first to maintain order later
    temp_results = {}
    
    # 3. Parallel Execution
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {executor.submit(process_single_question, q): q for q in questions}
        
        completed_count = 0
        for future in as_completed(future_to_question):
            q_text = future_to_question[future]
            try:
                res = future.result()
                if res:
                    temp_results[q_text] = res['analysis_result']
            except Exception as exc:
                print(f"Generated an exception for {q_text}: {exc}")
            
            completed_count += 1
            if completed_count % 2 == 0 or completed_count == len(questions):
                print(f"Progress: {completed_count}/{len(questions)} completed.")

    elapsed_time = time.time() - start_time
    print(f"Processing finished in {elapsed_time:.2f} seconds.")

    # 4. Format Output (Add IDs and structure)
    final_output = []
    current_id = 1

    # Loop through original questions list to preserve order
    for q in questions:
        if q in temp_results:
            entry = {
                "id": current_id,
                "original_question": q,
                "analysis_result": temp_results[q]
            }
            final_output.append(entry)
            current_id += 1
        else:
            print(f"[Warning] No result found for: {q}")

    # 5. Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"Done. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
