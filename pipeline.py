import json
import os
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure custom local dependencies are loaded
from classify_LTM import classify_text

# Define Pydantic schema based on extraction_prompt.txt
class DistilledKnowledge(BaseModel):
    decision: Literal["keep", "drop", "quarantine"]
    type: Literal["fact", "skill", "procedure", "heuristic", "anti_pattern", "certified_method"]
    domain: str
    statement: str
    procedure: List[str]
    constraints: List[str]
    confidence: float
    privacy_risk: Literal["low", "medium", "high"]
    reason: str

def main():
    # File Paths
    IN_FILE = "quarantine_raw_export.jsonl"
    OUT_FILE = "distilled_knowledge.jsonl"
    PROMPT_PATH = Path("extraction_prompt.txt")

    # 1. Load System Prompt
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: {PROMPT_PATH} not found.")
        return

    # 2. Initialize LLM Client
    # This expects the 'OPENAI_API_KEY' environment variable. 
    # The client can be redirected to local models by passing `base_url="http://localhost:11434/v1"` (e.g. for Ollama)
    client = OpenAI()
    # Replace with the actual model you intend to use
    MODEL_NAME = "gpt-4o" 

    processed_count = 0
    dropped_early = 0
    kept_count = 0
    
    if not os.path.exists(IN_FILE):
        print(f"Error: {IN_FILE} does not exist. Run extract_LTM.py first.")
        return

    print("Starting LTM Distillation Pipeline...\n")
    
    with open(IN_FILE, "r", encoding="utf-8") as f_in, \
         open(OUT_FILE, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            record = json.loads(line)
            doc_id = record.get("id", "unknown-id")
            doc_text = record.get("document", "")
            
            # Phase 1: Deterministic Filtering / Redaction
            classification = classify_text(doc_text)
            if classification == "DROP":
                dropped_early += 1
                print(f"[{doc_id}] Skipped by Regex filter.")
                continue
                
            print(f"[{doc_id}] Classified as {classification}. Offloading to LLM...")
            
            # Phase 2: LLM Knowledge Distillation
            try:
                # Using OpenAI structured output parsing
                response = client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract intelligence from this memory:\n{doc_text}"}
                    ],
                    response_format=DistilledKnowledge,
                    temperature=0.1
                )
                
                extracted = response.choices[0].message.parsed
                
                # Phase 3: Final programmatic Verification
                if extracted and extracted.decision == "keep" and extracted.privacy_risk == "low":
                    # Remove any potentially dangerous original metadata that isn't required
                    clean_metadata = {
                        "original_source_id": doc_id,
                    }
                    
                    clean_record = {
                        "id": f"distilled_{doc_id}",
                        "document": extracted.statement, # Using the exact clean statement
                        "metadata": {
                            **clean_metadata,
                            "type": extracted.type,
                            "domain": extracted.domain,
                            "confidence": extracted.confidence,
                            "procedure": extracted.procedure,
                            "constraints": extracted.constraints
                        }
                    }
                    f_out.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                    kept_count += 1
                    print(f"  -> Kept: {extracted.type} - {extracted.domain}")
                else:
                    reason = extracted.reason if extracted else "Failed parse"
                    level = extracted.privacy_risk if extracted else "unknown"
                    print(f"  -> Dropped by LLM (Privacy Risk: {level}). Reason: {reason}")
                    
            except Exception as e:
                print(f"  -> Error communicating with LLM for {doc_id}: {e}")
                
            processed_count += 1

    print("\n--- Pipeline Complete ---")
    print(f"  Evaluated via LLM  : {processed_count}")
    print(f"  Dropped by Regex   : {dropped_early}")
    print(f"  Successful Keeps   : {kept_count}")
    print(f"  Output saved to    : {OUT_FILE}")

if __name__ == "__main__":
    main()
