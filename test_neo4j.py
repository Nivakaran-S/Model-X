import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

print("="*60)
print("[NEO4J CONNECTION TEST]")
print("="*60)

user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password")

# Try multiple URIs
uris_to_test = [
    "bolt://localhost:7687",
    "neo4j://localhost:7687",
    "bolt://127.0.0.1:7687",
    "neo4j://127.0.0.1:7687"
]

success = False
for uri in uris_to_test:
    print(f"\nTrying: {uri}")
    print(f"User: {user}")
    print(f"Password: {'*' * len(password)}")
    
    try:
        driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            connection_timeout=10
        )
        driver.verify_connectivity()
        print(f"[SUCCESS] Connected with {uri}!")
        driver.close()
        success = True
        print(f"\n>> UPDATE .env with: NEO4J_URI={uri}")
        break
    except Exception as e:
        print(f"[FAILED] {type(e).__name__}: {e}")

if not success:
    print("\n[ERROR] All connection attempts failed!")
    print("Check if Docker container is running: docker ps | findstr neo4j")

print("="*60)
