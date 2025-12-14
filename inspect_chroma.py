
import logging
import sys
from pathlib import Path
from collections import Counter

# Setup path
sys.path.append(str(Path.cwd()))

from src.storage.config import config
from src.storage.chromadb_store import ChromaDBStore

# Mute logging
logging.basicConfig(level=logging.ERROR)

def inspect():
    print("Connecting to ChromaDB...")
    store = ChromaDBStore()
    
    if not store.collection:
        print("Could not connect to collection.")
        return

    count = store.collection.count()
    print(f"Total documents: {count}")
    
    if count == 0:
        return

    # Get a sample or all metadata
    # ChromaDB get() without ids returns everything if limit allows, or we can page.
    # tailored for 2000 docs, let's just get all metadatas.
    
    print("Fetching metadata...")
    data = store.collection.get(include=["metadatas"])
    metadatas = data["metadatas"]
    
    domains = Counter()
    sources = Counter()
    impacts = Counter()
    
    for meta in metadatas:
        if not meta: continue
        domains[meta.get("domain", "unknown")] += 1
        sources[meta.get("platform", "unknown")] += 1
        impacts[meta.get("impact_type", "unknown")] += 1
        
    print("\n--- Domain Distribution ---")
    for d, c in domains.most_common():
        print(f"{d}: {c}")
        
    print("\n--- Source/Platform Distribution ---")
    for s, c in sources.most_common():
        print(f"{s}: {c}")

    print("\n--- Impact Type Distribution ---")
    for i, c in impacts.most_common():
        print(f"{i}: {c}")

if __name__ == "__main__":
    inspect()
