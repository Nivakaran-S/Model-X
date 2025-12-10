"""
src/rag.py
Chat-History Aware RAG Application for Roger Intelligence Platform
Connects to all ChromaDB collections used by the agent graph for conversational Q&A.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("Roger_rag")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ============================================
# IMPORTS
# ============================================

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("[RAG] ChromaDB not available. Install with: pip install chromadb")

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning(
        "[RAG] LangChain not available. Install with: pip install langchain-groq langchain-core"
    )


# ============================================
# CHROMADB MULTI-COLLECTION RETRIEVER
# ============================================


class MultiCollectionRetriever:
    """
    Connects to all ChromaDB collections used by Roger agents.
    Provides unified search across all intelligence data.
    """

    # Known collections from the agents
    COLLECTIONS = [
        "Roger_feeds",  # From chromadb_store.py (storage manager)
        "Roger_rag_collection",  # From db_manager.py (agent nodes)
    ]

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or os.getenv(
            "CHROMADB_PATH", str(PROJECT_ROOT / "data" / "chromadb")
        )
        self.client = None
        self.collections: Dict[str, Any] = {}

        if not CHROMA_AVAILABLE:
            logger.error("[RAG] ChromaDB not installed!")
            return

        self._init_client()

    def _init_client(self):
        """Initialize ChromaDB client and connect to all collections"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # List all available collections
            all_collections = self.client.list_collections()
            available_names = [c.name for c in all_collections]

            logger.info(
                f"[RAG] Found {len(all_collections)} collections: {available_names}"
            )

            # Connect to known collections
            for name in self.COLLECTIONS:
                if name in available_names:
                    self.collections[name] = self.client.get_collection(name)
                    count = self.collections[name].count()
                    logger.info(f"[RAG] ✓ Connected to '{name}' ({count} documents)")

            # Also connect to any other collections found
            for name in available_names:
                if name not in self.collections:
                    self.collections[name] = self.client.get_collection(name)
                    count = self.collections[name].count()
                    logger.info(f"[RAG] ✓ Connected to '{name}' ({count} documents)")

            if not self.collections:
                logger.warning(
                    "[RAG] No collections found! Agents may not have stored data yet."
                )

        except Exception as e:
            logger.error(f"[RAG] ChromaDB initialization error: {e}")
            self.client = None

    def search(
        self, query: str, n_results: int = 5, domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search across all collections for relevant documents.

        Args:
            query: Search query
            n_results: Max results per collection
            domain_filter: Optional domain to filter (political, economic, weather, social)

        Returns:
            List of results with metadata
        """
        if not self.client:
            return []

        all_results = []

        for name, collection in self.collections.items():
            try:
                # Build where filter if domain specified
                where_filter = None
                if domain_filter:
                    where_filter = {"domain": domain_filter.lower()}

                results = collection.query(
                    query_texts=[query], n_results=n_results, where=where_filter
                )

                # Process results
                if results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        doc = results["documents"][0][i] if results["documents"] else ""
                        meta = (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        )
                        distance = (
                            results["distances"][0][i] if results["distances"] else 0
                        )

                        # Calculate similarity score
                        similarity = 1.0 - min(distance / 2.0, 1.0)

                        all_results.append(
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

        # Sort by similarity (highest first)
        all_results.sort(key=lambda x: x["similarity"], reverse=True)

        return all_results[: n_results * 2]  # Return top results across all collections

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
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


# ============================================
# CHAT-HISTORY AWARE RAG CHAIN
# ============================================


class RogerRAG:
    """
    Chat-history aware RAG for Roger Intelligence Platform.
    Uses Groq LLM and multi-collection ChromaDB retrieval.
    """

    def __init__(self):
        self.retriever = MultiCollectionRetriever()
        self.llm = None
        self.chat_history: List[Tuple[str, str]] = []

        if LANGCHAIN_AVAILABLE:
            self._init_llm()

    def _init_llm(self):
        """Initialize Groq LLM"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.error("[RAG] GROQ_API_KEY not set!")
                return

            self.llm = ChatGroq(
                api_key=api_key,
                model="openai/gpt-oss-120b",  # Good for RAG
                temperature=0.3,
                max_tokens=1024,
            )
            logger.info("[RAG] ✓ Groq LLM initialized (OpenAI/gpt-oss-120b)")

        except Exception as e:
            logger.error(f"[RAG] LLM initialization error: {e}")

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for LLM with temporal awareness"""
        if not docs:
            return "No relevant intelligence data found."

        context_parts = []
        now = datetime.now()

        for i, doc in enumerate(docs[:5], 1):  # Top 5 docs
            meta = doc.get("metadata", {})
            domain = meta.get("domain", "unknown")
            platform = meta.get("platform", "")
            timestamp = meta.get("timestamp", "")

            # Calculate age of the source
            age_str = "unknown date"
            if timestamp:
                try:
                    # Try to parse various timestamp formats
                    for fmt in [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d",
                        "%d/%m/%Y",
                    ]:
                        try:
                            ts_date = datetime.strptime(timestamp[:19], fmt)
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
                                age_str = f"{days_old // 30} months ago (⚠️ POTENTIALLY OUTDATED)"
                            else:
                                age_str = f"{days_old // 365} years ago (⚠️ OUTDATED)"
                            break
                        except ValueError:
                            continue
                except Exception:
                    age_str = f"Date: {timestamp}"

            context_parts.append(
                f"[Source {i}] Domain: {domain} | Platform: {platform}\n"
                f"📅 TIMESTAMP: {timestamp} ({age_str})\n"
                f"{doc['content']}\n"
            )

        return "\n---\n".join(context_parts)

    def _reformulate_question(self, question: str) -> str:
        """Reformulate question using chat history for context"""
        if not self.chat_history or not self.llm:
            return question

        # Build history context
        history_text = ""
        for human, ai in self.chat_history[-3:]:  # Last 3 exchanges
            history_text += f"Human: {human}\nAssistant: {ai}\n"

        # Create reformulation prompt
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
        """
        Query the RAG system with chat-history awareness.

        Args:
            question: User's question
            domain_filter: Optional domain filter (political, economic, weather, social, intelligence)
            use_history: Whether to use chat history for context

        Returns:
            Dict with answer, sources, and metadata
        """
        # Reformulate question if we have history
        search_question = question
        if use_history and self.chat_history:
            search_question = self._reformulate_question(question)

        # Retrieve relevant documents
        docs = self.retriever.search(
            search_question, n_results=5, domain_filter=domain_filter
        )

        if not docs:
            return {
                "answer": "I couldn't find any relevant intelligence data to answer your question. The agents may not have collected data yet, or your question might need different keywords.",
                "sources": [],
                "question": question,
                "reformulated": (
                    search_question if search_question != question else None
                ),
            }

        # Format context
        context = self._format_context(docs)

        # Generate answer
        if not self.llm:
            return {
                "answer": f"LLM not available. Here's the raw context:\n\n{context}",
                "sources": docs,
                "question": question,
            }

        # RAG prompt with temporal awareness
        current_date = datetime.now().strftime("%B %d, %Y")
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are Roger, an AI intelligence analyst for Sri Lanka.
            
TODAY'S DATE: {current_date}

CRITICAL TEMPORAL AWARENESS INSTRUCTIONS:
1. ALWAYS check the timestamp/date of each source before using information
2. For questions about "current" situations, ONLY use sources from the last 30 days
3. If sources are outdated (more than 30 days old), explicitly mention this: "Based on data from [date], which may be outdated..."
4. For political leadership questions, verify information is from recent sources
5. If you find conflicting information from different time periods, prefer the most recent source
6. Never present old information as current fact without temporal qualification

IMPORTANT POLITICAL CONTEXT:
- Presidential elections were held in Sri Lanka in September 2024
- Always verify any claims about political leadership against the most recent sources

Answer questions based ONLY on the provided intelligence context.
Be concise but informative. Always cite source timestamps when available.
If the context doesn't contain relevant RECENT information for current-state questions, say so.
            
Context (check timestamps carefully):
{{context}}""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # Build history messages
        history_messages = []
        for human, ai in self.chat_history[-5:]:  # Last 5 exchanges
            history_messages.append(HumanMessage(content=human))
            history_messages.append(AIMessage(content=ai))

        try:
            chain = rag_prompt | self.llm | StrOutputParser()
            answer = chain.invoke(
                {"context": context, "history": history_messages, "question": question}
            )

            # Update chat history
            self.chat_history.append((question, answer))

            # Prepare sources summary
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
        """Clear chat history"""
        self.chat_history = []
        logger.info("[RAG] Chat history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "retriever": self.retriever.get_stats(),
            "llm_available": self.llm is not None,
            "chat_history_length": len(self.chat_history),
        }


# ============================================
# CLI INTERFACE
# ============================================


def run_cli():
    """Interactive CLI for testing the RAG system"""
    print("\n" + "=" * 60)
    print("  🇱🇰 Roger Intelligence RAG")
    print("  Chat-History Aware Q&A System")
    print("=" * 60)

    rag = RogerRAG()

    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 Connected Collections: {stats['retriever']['total_collections']}")
    print(f"📄 Total Documents: {stats['retriever']['total_documents']}")
    print(f"🤖 LLM Available: {'Yes' if stats['llm_available'] else 'No'}")

    if stats["retriever"]["total_documents"] == 0:
        print("\n⚠️  No documents found! Make sure the agents have collected data.")

    print("\nCommands:")
    print("  /clear  - Clear chat history")
    print("  /stats  - Show system statistics")
    print("  /domain <name> - Filter by domain (political, economic, weather, social)")
    print("  /quit   - Exit")
    print("-" * 60)

    domain_filter = None

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == "/clear":
                rag.clear_history()
                print("✓ Chat history cleared")
                continue

            if user_input.lower() == "/stats":
                print(f"\n📊 Stats: {rag.get_stats()}")
                continue

            if user_input.lower().startswith("/domain"):
                parts = user_input.split()
                if len(parts) > 1:
                    domain_filter = parts[1] if parts[1] != "all" else None
                    print(f"✓ Domain filter: {domain_filter or 'all'}")
                else:
                    print("Usage: /domain <political|economic|weather|social|all>")
                continue

            # Query RAG
            print("\n🔍 Searching intelligence database...")
            result = rag.query(user_input, domain_filter=domain_filter)

            # Show answer
            print(f"\n🤖 Roger: {result['answer']}")

            # Show sources
            if result.get("sources"):
                print(f"\n📚 Sources ({len(result['sources'])} found):")
                for i, src in enumerate(result["sources"][:3], 1):
                    print(
                        f"   {i}. {src['domain']} | {src['platform']} | Relevance: {src['similarity']:.0%}"
                    )

            if result.get("reformulated"):
                print(f"\n💡 (Interpreted as: {result['reformulated']})")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    run_cli()
