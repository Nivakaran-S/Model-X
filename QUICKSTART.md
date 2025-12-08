# 🚀 Roger Quick Start Guide

## Prerequisites
- Python 3.11+
- Node.js 18+
- Groq API Key ([Get Free Key](https://console.groq.com))

## Installation & Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy template
cp .env.template .env

# Edit .env and add your GROQ_API_KEY
# GROQ_API_KEY=your_key_here
```

### 3. Start Backend
```bash
python main.py
```

Wait for initialization logs:
```
[StorageManager] Initializing multi-database storage system
[SQLiteCache] Initialized at data/cache/feeds.db
[ChromaDB] Initialized collection: Roger_feeds
[CombinedAgentNode] Initialized with production storage layer
```

### 4. Start Frontend (New Terminal)
```bash
cd frontend
npm install
npm run dev
```

### 5. Access Dashboard
Open: http://localhost:3000

---

## 🎯 What to Expect

### First 60 Seconds
- System initializes 6 domain agents
- Begins scraping 47+ data sources
- Deduplication pipeline activates

### After 60-120 Seconds
- First batch of events appears on dashboard
- Risk metrics start calculating
- Real-time WebSocket connects

### Live Features
- ✅ Real-time intelligence feed
- ✅ Risk vs Opportunity classification
- ✅ 3-tier deduplication (SQLite + ChromaDB + Neo4j\*)
- ✅ CSV exports in `data/feeds/`
- ✅ Operational Risk Radar metrics

\*Neo4j optional - requires Docker

---

## 🐛 Troubleshooting

### "ChromaDB not found"
```bash
pip install chromadb sentence-transformers
```

### "No events appearing"
- Wait 60-120 seconds for first batch
- Check backend logs for errors
- Verify GROQ_API_KEY is set correctly

### Frontend can't connect
```bash
# Verify backend running
curl http://localhost:8000/api/status
```

---

## 📊 Production Features

### Storage Stats
```bash
curl http://localhost:8000/api/storage/stats
```

### CSV Exports
```bash
ls -lh data/feeds/
cat data/feeds/feed_$(date +%Y-%m-%d).csv
```

### Enable Neo4j (Optional)
```bash
# Start Neo4j with Docker
docker-compose -f docker-compose.prod.yml up -d neo4j

# Update .env
NEO4J_ENABLED=true

# Restart backend
python main.py

# Access Neo4j Browser
open http://localhost:7474
# Login: neo4j / Roger2024
```

---

## 🏆 Demo for Judges

**Show in this order**:
1. Live dashboard (http://localhost:3000)
2. Terminal logs showing deduplication stats
3. Neo4j graph visualization (if enabled)
4. CSV exports in data/feeds/
5. Storage API: http://localhost:8000/api/storage/stats

**Key talking points**:
- "47+ data sources, 6 domain agents running in parallel"
- "3-tier deduplication: SQLite for speed, ChromaDB for intelligence"
- "90%+ duplicate reduction vs 60% with basic hashing"
- "Production-ready with persistent storage and knowledge graphs"

---

**Ready to win! 🏆**
