import json
from pathlib import Path
import chromadb

OLD_CHROMA_PATH = "./Max_Botnick_Memory_May15/chroma_db"
COLLECTION_NAME = "memories"
OUT = Path("max_quarantine_raw_export.jsonl")

client = chromadb.PersistentClient(path=OLD_CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

batch_size = 500
total = collection.count()

with OUT.open("w", encoding="utf-8") as f:
    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]  # deliberately exclude embeddings
        )

        for rid, doc, meta in zip(
            batch.get("ids", []),
            batch.get("documents", []),
            batch.get("metadatas", [])
        ):
            f.write(json.dumps({
                "id": rid,
                "document": doc,
                "metadata": meta or {}
            }, ensure_ascii=False) + "\n")


