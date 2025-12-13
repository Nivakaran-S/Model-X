"""
rag.py - Chat-History Aware RAG Application for Roger Intelligence Platform
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("Roger_rag")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("[RAG] ChromaDB not available")

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("[RAG] LangChain not available")

# Neo4j for graph-based retrieval
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("[RAG] Neo4j not available")


# Keywords that indicate a graph/relationship query
GRAPH_KEYWORDS = [
    "connected",
    "related",
    "timeline",
    "before",
    "after",
    "caused by",
    "followed by",
    "similar to",
    "linked",
    "what happened",
    "sequence",
    "chain of events",
]


def is_graph_query(question: str) -> bool:
    """Detect if question requires graph traversal."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in GRAPH_KEYWORDS)


class Neo4jRetriever:
    """Graph-based retrieval for relationship queries with LAZY initialization."""

    def __init__(self):
        self.driver = None
        self._initialized = False
        self._init_attempted = False

    def _lazy_init(self):
        """Lazy initialization - only connect when actually needed."""
        if self._init_attempted:
            return self.driver is not None

        self._init_attempted = True

        if not NEO4J_AVAILABLE:
            logger.info("[Neo4jRetriever] Neo4j package not installed")
            return False

        neo4j_uri = os.getenv("NEO4J_URI", "")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "")

        if not neo4j_uri or not neo4j_password:
            logger.info("[Neo4jRetriever] Neo4j credentials not configured - skipping")
            return False

        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            self.driver.verify_connectivity()
            self._initialized = True
            logger.info(f"[Neo4jRetriever] Connected to {neo4j_uri}")
            return True
        except Exception as e:
            logger.warning(
                f"[Neo4jRetriever] Connection failed (will skip graph queries): {e}"
            )
            self.driver = None
            return False

    def get_related_events(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find events containing keyword and their related events."""
        if not self._lazy_init():
            return []

        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Event)
                WHERE toLower(e.summary) CONTAINS toLower($keyword)
                OPTIONAL MATCH (e)-[:SIMILAR_TO]-(related:Event)
                RETURN e.event_id as event_id,
                       e.summary as summary,
                       e.domain as domain,
                       e.severity as severity,
                       e.timestamp as timestamp,
                       COLLECT(DISTINCT related.summary)[0..3] as related_summaries
                ORDER BY e.timestamp DESC
                LIMIT $limit
                """
                results = session.run(query, keyword=keyword, limit=limit)

                events = []
                for record in results:
                    events.append(
                        {
                            "event_id": record["event_id"],
                            "content": record["summary"],
                            "domain": record["domain"],
                            "severity": record["severity"],
                            "timestamp": record["timestamp"],
                            "related": record["related_summaries"],
                            "source": "neo4j_graph",
                        }
                    )

                logger.info(
                    f"[Neo4jRetriever] Found {len(events)} events for '{keyword}'"
                )
                return events

        except Exception as e:
            logger.error(f"[Neo4jRetriever] Query error: {e}")
            return []

    def get_domain_events(self, domain: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent events by domain with relationships."""
        if not self._lazy_init():
            return []

        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Event)-[:BELONGS_TO]->(d:Domain {name: $domain})
                OPTIONAL MATCH (e)-[:SIMILAR_TO]-(related:Event)
                RETURN e.event_id as event_id,
                       e.summary as summary,
                       e.severity as severity,
                       e.timestamp as timestamp,
                       COUNT(related) as related_count
                ORDER BY e.timestamp DESC
                LIMIT $limit
                """
                results = session.run(query, domain=domain.lower(), limit=limit)

                events = []
                for record in results:
                    events.append(
                        {
                            "event_id": record["event_id"],
                            "content": record["summary"],
                            "domain": domain,
                            "severity": record["severity"],
                            "timestamp": record["timestamp"],
                            "related_count": record["related_count"],
                            "source": "neo4j_graph",
                        }
                    )

                return events

        except Exception as e:
            logger.error(f"[Neo4jRetriever] Domain query error: {e}")
            return []

    def get_event_chain(self, keyword: str, depth: int = 3) -> List[Dict[str, Any]]:
        """Get temporal chain of related events."""
        if not self._lazy_init():
            return []

        try:
            with self.driver.session() as session:
                query = """
                MATCH (start:Event)
                WHERE toLower(start.summary) CONTAINS toLower($keyword)
                OPTIONAL MATCH path = (start)-[:FOLLOWS|SIMILAR_TO*1..3]-(chain:Event)
                WITH start, COLLECT(DISTINCT chain) as chain_events
                RETURN start.event_id as start_id,
                       start.summary as start_summary,
                       start.timestamp as start_time,
                       [e IN chain_events | {summary: e.summary, time: e.timestamp}][0..5] as chain
                LIMIT 1
                """
                result = session.run(query, keyword=keyword).single()

                if result:
                    return [
                        {
                            "event_id": result["start_id"],
                            "content": result["start_summary"],
                            "timestamp": result["start_time"],
                            "chain": result["chain"],
                            "source": "neo4j_chain",
                        }
                    ]
                return []

        except Exception as e:
            logger.error(f"[Neo4jRetriever] Chain query error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get Neo4j graph statistics."""
        if not self._initialized or not self.driver:
            return {
                "status": (
                    "not_initialized" if not self._init_attempted else "disconnected"
                )
            }

        try:
            with self.driver.session() as session:
                event_count = session.run(
                    "MATCH (e:Event) RETURN COUNT(e) as count"
                ).single()["count"]

                return {"status": "connected", "total_events": event_count}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class MultiCollectionRetriever:
    COLLECTIONS = ["Roger_feeds"]

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or os.getenv(
            "CHROMADB_PATH", str(PROJECT_ROOT / "data" / "chromadb")
        )
        self.client = None
        self.collections: Dict[str, Any] = {}

        # Thread pool for parallel queries
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=4)

        if not CHROMA_AVAILABLE:
            logger.error("[RAG] ChromaDB not installed")
            return

        self._init_client()

    def _init_client(self):
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            all_collections = self.client.list_collections()
            available_names = [c.name for c in all_collections]

            logger.info(
                f"[RAG] Found {len(all_collections)} collections: {available_names}"
            )

            for name in self.COLLECTIONS:
                if name in available_names:
                    self.collections[name] = self.client.get_collection(name)
                    count = self.collections[name].count()
                    logger.info(f"[RAG] Connected to '{name}' ({count} documents)")

            for name in available_names:
                if name not in self.collections:
                    self.collections[name] = self.client.get_collection(name)
                    count = self.collections[name].count()
                    logger.info(f"[RAG] Connected to '{name}' ({count} documents)")

            if not self.collections:
                logger.warning("[RAG] No collections found")

        except Exception as e:
            logger.error(f"[RAG] ChromaDB initialization error: {e}")
            self.client = None

    def _query_single_collection(
        self,
        name: str,
        collection,
        query: str,
        n_results: int,
        domain_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Query a single collection - used for parallel execution."""
        results_list = []
        try:
            where_filter = None
            if domain_filter:
                where_filter = {"domain": domain_filter.lower()}

            results = collection.query(
                query_texts=[query], n_results=n_results, where=where_filter
            )

            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = results["documents"][0][i] if results["documents"] else ""
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0

                    similarity = 1.0 - min(distance / 2.0, 1.0)

                    results_list.append(
                        {
                            "id": doc_id,
                            "content": doc,
                            "metadata": meta,
                            "similarity": similarity,
                            "collection": name,
                            "domain": meta.get("domain", "unknown"),
                        }
                    )

        except Exception as e:
            logger.warning(f"[RAG] Error querying {name}: {e}")

        return results_list

    def search(
        self, query: str, n_results: int = 5, domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search all collections in PARALLEL for faster results."""
        if not self.client:
            return []

        # Submit parallel queries to all collections
        from concurrent.futures import as_completed

        futures = {}
        for name, collection in self.collections.items():
            future = self._executor.submit(
                self._query_single_collection,
                name,
                collection,
                query,
                n_results,
                domain_filter,
            )
            futures[future] = name

        # Collect results as they complete (fastest first)
        all_results = []
        for future in as_completed(futures, timeout=10.0):  # 10s timeout
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.warning(
                    f"[RAG] Parallel query failed for {futures[future]}: {e}"
                )

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[: n_results * 2]

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "total_collections": len(self.collections),
            "total_documents": 0,
            "collections": {},
        }

        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats["collections"][name] = count
                stats["total_documents"] += count
            except Exception:
                stats["collections"][name] = "error"

        return stats


class RogerRAG:
    def __init__(self):
        self.retriever = MultiCollectionRetriever()
        # Neo4j disabled for faster startup - uncomment when graph DB is configured
        # self.neo4j_retriever = Neo4jRetriever()  # Graph-based retrieval
        self.neo4j_retriever = None  # Disabled
        self.llm = None
        self.chat_history: List[Tuple[str, str]] = []

        if LANGCHAIN_AVAILABLE:
            self._init_llm()

    def _init_llm(self):
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.error("[RAG] GROQ_API_KEY not set")
                return

            # Using Llama 4 Maverick 17B for fast, high-quality responses
            self.llm = ChatGroq(
                api_key=api_key,
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.3,
                max_tokens=1024,
                request_timeout=30,  # 30 second timeout
            )
            logger.info("[RAG] Groq LLM initialized with Llama 4 Maverick 17B")

        except Exception as e:
            logger.error(f"[RAG] LLM initialization error: {e}")

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract key terms from question for graph search."""
        # Remove common stopwords
        stopwords = {
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "about",
            "related",
            "connected",
            "happened",
            "after",
            "before",
            "show",
            "me",
            "tell",
            "find",
            "get",
            "events",
            "timeline",
        }

        words = question.lower().replace("?", "").replace(",", "").split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        return keywords[:5]  # Return top 5 keywords

    def _format_context(
        self, docs: List[Dict[str, Any]], include_graph: bool = False
    ) -> str:
        if not docs:
            return "No relevant intelligence data found."

        context_parts = []
        now = datetime.now()

        # Separate ChromaDB and Neo4j results
        chroma_docs = [d for d in docs if d.get("source") != "neo4j_graph"]
        graph_docs = [d for d in docs if d.get("source") == "neo4j_graph"]

        # Format ChromaDB results
        for i, doc in enumerate(chroma_docs[:5], 1):
            meta = doc.get("metadata", {})
            domain = meta.get("domain", doc.get("domain", "unknown"))
            platform = meta.get("platform", "")
            timestamp = meta.get("timestamp", doc.get("timestamp", ""))

            age_str = "unknown date"
            if timestamp:
                try:
                    for fmt in [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d",
                        "%d/%m/%Y",
                    ]:
                        try:
                            ts_date = datetime.strptime(str(timestamp)[:19], fmt)
                            days_old = (now - ts_date).days
                            if days_old == 0:
                                age_str = "TODAY"
                            elif days_old == 1:
                                age_str = "1 day ago"
                            elif days_old < 7:
                                age_str = f"{days_old} days ago"
                            elif days_old < 30:
                                age_str = f"{days_old // 7} weeks ago"
                            elif days_old < 365:
                                age_str = f"{days_old // 30} months ago (POTENTIALLY OUTDATED)"
                            else:
                                age_str = f"{days_old // 365} years ago (OUTDATED)"
                            break
                        except ValueError:
                            continue
                except Exception:
                    age_str = f"Date: {timestamp}"

            context_parts.append(
                f"[Source {i}] Domain: {domain} | Platform: {platform}\n"
                f"TIMESTAMP: {timestamp} ({age_str})\n"
                f"{doc['content']}\n"
            )

        # Format Neo4j graph results (if any)
        if graph_docs:
            context_parts.append("\n=== RELATED EVENTS FROM KNOWLEDGE GRAPH ===\n")
            for i, doc in enumerate(graph_docs[:3], 1):
                related = doc.get("related", [])
                related_str = ""
                if related:
                    related_str = f"\n  Related events: {', '.join(str(r)[:50] + '...' for r in related[:2])}"

                context_parts.append(
                    f"[Graph {i}] Domain: {doc.get('domain', 'unknown')} | "
                    f"Severity: {doc.get('severity', 'unknown')}\n"
                    f"{doc.get('content', '')[:500]}"
                    f"{related_str}\n"
                )

        return "\n---\n".join(context_parts)

    def _reformulate_question(self, question: str) -> str:
        if not self.chat_history or not self.llm:
            return question

        history_text = ""
        for human, ai in self.chat_history[-3:]:
            history_text += f"Human: {human}\nAssistant: {ai}\n"

        reformulate_prompt = ChatPromptTemplate.from_template(
            """Given the following conversation history and a follow-up question, 
            reformulate the follow-up question to be a standalone question that captures the full context.
            
            Chat History:
            {history}
            
            Follow-up Question: {question}
            
            Standalone Question:"""
        )

        try:
            chain = reformulate_prompt | self.llm | StrOutputParser()
            standalone = chain.invoke({"history": history_text, "question": question})
            logger.info(f"[RAG] Reformulated: '{question}' -> '{standalone.strip()}'")
            return standalone.strip()
        except Exception as e:
            logger.warning(f"[RAG] Reformulation failed: {e}")
            return question

    def query(
        self,
        question: str,
        domain_filter: Optional[str] = None,
        use_history: bool = True,
    ) -> Dict[str, Any]:
        search_question = question
        if use_history and self.chat_history:
            search_question = self._reformulate_question(question)

        # ChromaDB semantic search (always)
        docs = self.retriever.search(
            search_question, n_results=5, domain_filter=domain_filter
        )

        # Neo4j graph search (for relationship queries) - only if enabled
        graph_docs = []
        used_graph = False
        if self.neo4j_retriever and is_graph_query(search_question):
            logger.info(f"[RAG] Graph query detected: '{search_question}'")
            used_graph = True

            # Extract keywords for graph search
            # Simple: use first nouns/keywords from question
            keywords = self._extract_keywords(search_question)

            for keyword in keywords[:2]:  # Limit to 2 keywords
                graph_docs.extend(
                    self.neo4j_retriever.get_related_events(keyword, limit=3)
                )

            logger.info(f"[RAG] Graph retrieval: {len(graph_docs)} docs from Neo4j")

        # Merge results (ChromaDB + Neo4j)
        all_docs = docs + graph_docs

        if not all_docs:
            return {
                "answer": "I couldn't find any relevant intelligence data to answer your question.",
                "sources": [],
                "question": question,
                "reformulated": (
                    search_question if search_question != question else None
                ),
            }

        context = self._format_context(all_docs, include_graph=used_graph)

        if not self.llm:
            return {
                "answer": f"LLM not available. Here's the raw context:\n\n{context}",
                "sources": docs,
                "question": question,
            }

        current_date = datetime.now().strftime("%B %d, %Y")
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are Roger, an AI intelligence analyst for Sri Lanka.
            
TODAY'S DATE: {current_date}

TEMPORAL AWARENESS INSTRUCTIONS:
1. Check the timestamp/date of each source before using information
2. For questions about "current" situations, prefer sources from the last 30 days
3. If sources are outdated, mention this explicitly
4. For political leadership questions, verify information is from recent sources
5. Never present old information as current fact without temporal qualification
6. Never use tables to answers.. Your answers should always be a paragraph or in bullet points

Answer questions based ONLY on the provided intelligence context.
Be concise but informative. Cite source timestamps when available.
            
Context:
{{context}}""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        history_messages = []
        for human, ai in self.chat_history[-5:]:
            history_messages.append(HumanMessage(content=human))
            history_messages.append(AIMessage(content=ai))

        try:
            chain = rag_prompt | self.llm | StrOutputParser()
            answer = chain.invoke(
                {"context": context, "history": history_messages, "question": question}
            )

            self.chat_history.append((question, answer))

            sources_summary = []
            for doc in docs[:5]:
                meta = doc.get("metadata", {})
                sources_summary.append(
                    {
                        "domain": meta.get("domain", "unknown"),
                        "platform": meta.get("platform", "unknown"),
                        "category": meta.get("category", ""),
                        "similarity": round(doc["similarity"], 3),
                    }
                )

            return {
                "answer": answer,
                "sources": sources_summary,
                "question": question,
                "reformulated": (
                    search_question if search_question != question else None
                ),
                "docs_found": len(docs),
            }

        except Exception as e:
            logger.error(f"[RAG] Query error: {e}")
            return {
                "answer": f"Error generating response: {e}",
                "sources": [],
                "question": question,
                "error": str(e),
            }

    def clear_history(self):
        self.chat_history = []
        logger.info("[RAG] Chat history cleared")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "retriever": self.retriever.get_stats(),
            "llm_available": self.llm is not None,
            "chat_history_length": len(self.chat_history),
        }


def run_cli():
    print("Roger Intelligence RAG - Chat-History Aware Q&A System")

    rag = RogerRAG()
    stats = rag.get_stats()
    print(f"Connected Collections: {stats['retriever']['total_collections']}")
    print(f"Total Documents: {stats['retriever']['total_documents']}")
    print(f"LLM Available: {'Yes' if stats['llm_available'] else 'No'}")

    if stats["retriever"]["total_documents"] == 0:
        print("No documents found. Make sure the agents have collected data.")

    print("\nCommands: /clear, /stats, /domain <name>, /quit")

    domain_filter = None

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/clear":
                rag.clear_history()
                print("Chat history cleared")
                continue

            if user_input.lower() == "/stats":
                print(f"Stats: {rag.get_stats()}")
                continue

            if user_input.lower().startswith("/domain"):
                parts = user_input.split()
                if len(parts) > 1:
                    domain_filter = parts[1] if parts[1] != "all" else None
                    print(f"Domain filter: {domain_filter or 'all'}")
                else:
                    print("Usage: /domain <political|economic|weather|social|all>")
                continue

            print("Searching intelligence database...")
            result = rag.query(user_input, domain_filter=domain_filter)

            print(f"\nRoger: {result['answer']}")

            if result.get("sources"):
                print(f"\nSources ({len(result['sources'])} found):")
                for i, src in enumerate(result["sources"][:3], 1):
                    print(
                        f"   {i}. {src['domain']} | {src['platform']} | Relevance: {src['similarity']:.0%}"
                    )

            if result.get("reformulated"):
                print(f"\n(Interpreted as: {result['reformulated']})")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_cli()
