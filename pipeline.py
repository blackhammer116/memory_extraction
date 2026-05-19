import json
import os
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import tqdm

# Load environment variables from .env file
load_dotenv()

# Ensure custom local dependencies are loaded
from classify_LTM import classify_text

# Define Pydantic schema based on extraction_prompt.txt
class DistilledKnowledge(BaseModel):
    reference_id: str
    decision: Literal["keep", "drop", "quarantine"]
    type: Literal["fact", "skill", "procedure", "heuristic", "anti_pattern", "certified_method"]
    domain: str
    statement: str
    procedure: List[str]
    constraints: List[str]
    confidence: float
    privacy_risk: Literal["low", "medium", "high"]
    reason: str

class BatchDistilledKnowledge(BaseModel):
    items: List[DistilledKnowledge]


def main():
    # File Paths
    IN_FILE = "max_quarantine_raw_export.jsonl"
    OUT_FILE = "KB/max_distilled_knowledge.jsonl"
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

    with open(IN_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())

    print("Starting LTM Distillation Pipeline...\n")
    
    BATCH_SIZE = 10  # Process 10 memories per LLM call
    batch_records = []
    
    with open(IN_FILE, "r", encoding="utf-8") as f_in, \
         open(OUT_FILE, "w", encoding="utf-8") as f_out:
        
        progress_bar = tqdm.tqdm(f_in, total=total_lines, desc="Processing Memories")
        
        def process_batch(batch):
            nonlocal processed_count, kept_count
            if not batch:
                return
        
            batch_text = "\n\n".join([f"<memory id=\"{rec['id']}\">\n{rec['document']}\n</memory>" for rec in batch])
            
            try:
                response = client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract intelligence for each of the following memories. Make sure to map the reference_id to the memory id attribute:\n\n{batch_text}"}
                    ],
                    response_format=BatchDistilledKnowledge,
                    temperature=0.1
                )
                
                extracted_batch = response.choices[0].message.parsed.items
                
                # Process results
                for extracted in extracted_batch:
                    # Find original record text if needed (optional)
                    # orig_record = next((r for r in batch if r['id'] == extracted.reference_id), None)
                    
                    if extracted.decision == "keep" and extracted.privacy_risk == "low":
                        clean_metadata = {
                            "original_source_id": extracted.reference_id,
                        }
                        
                        clean_record = {
                            "id": f"distilled_{extracted.reference_id}",
                            "document": extracted.statement,
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
                        tqdm.tqdm.write(f"  -> Kept: [{extracted.reference_id}] {extracted.type} - {extracted.domain}")
                    else:
                        tqdm.tqdm.write(f"  -> Dropped by LLM [{extracted.reference_id}] (Risk: {extracted.privacy_risk}). Reason: {extracted.reason}")
                        
            except Exception as e:
                tqdm.tqdm.write(f"  -> Error communicating with LLM for batch: {e}")
                
            processed_count += len(batch)

        for line in progress_bar:
            if not line.strip():
                continue
                
            record = json.loads(line)
            doc_id = record.get("id", "unknown-id")
            doc_text = record.get("document", "")
            
            # Phase 1: Deterministic Filtering / Redaction
            classification = classify_text(doc_text)
            if classification == "DROP":
                dropped_early += 1
                progress_bar.write(f"[{doc_id}] Skipped by Regex filter.")
                continue
                
            # Queue for batch processing
            batch_records.append({"id": doc_id, "document": doc_text})
            
            if len(batch_records) >= BATCH_SIZE:
                process_batch(batch_records)
                batch_records.clear()
                
        # Process any remaining records
        if batch_records:
            process_batch(batch_records)

    print("\n--- Pipeline Complete ---")
    print(f"  Evaluated via LLM  : {processed_count}")
    print(f"  Dropped by Regex   : {dropped_early}")
    print(f"  Successful Keeps   : {kept_count}")
    print(f"  Output saved to    : {OUT_FILE}")

if __name__ == "__main__":
    main()
