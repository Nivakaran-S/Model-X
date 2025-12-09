"""
src/graphs/vectorizationAgentGraph.py
Vectorization Agent Graph - Agentic workflow for text-to-vector conversion
"""

from langgraph.graph import StateGraph, END
from src.states.vectorizationAgentState import VectorizationAgentState
from src.nodes.vectorizationAgentNode import VectorizationAgentNode
from src.llms.groqllm import GroqLLM


class VectorizationGraphBuilder:
    """
    Builds the Vectorization Agent graph.

    Architecture (Sequential Pipeline):
    Step 1: Language Detection (FastText/lingua-py)
    Step 2: Text Vectorization (SinhalaBERTo/Tamil-BERT/DistilBERT)
    Step 3: Anomaly Detection (Isolation Forest on vectors)
    Step 4: Trending Detection (Velocity/Spike tracking)
    Step 5: Expert Summary (GroqLLM)
    Step 6: Format Output
    """

    def __init__(self, llm=None):
        self.llm = llm or GroqLLM().get_llm()

    def build_graph(self):
        """
        Build the vectorization agent graph.

        Flow:
        detect_languages → vectorize_texts → anomaly_detection → trending_detection → expert_summary → format_output → END
        """
        node = VectorizationAgentNode(self.llm)

        # Create graph
        graph = StateGraph(VectorizationAgentState)

        # Add nodes
        graph.add_node("detect_languages", node.detect_languages)
        graph.add_node("vectorize_texts", node.vectorize_texts)
        graph.add_node("anomaly_detection", node.run_anomaly_detection)
        graph.add_node("trending_detection", node.run_trending_detection)
        graph.add_node("generate_expert_summary", node.generate_expert_summary)
        graph.add_node("format_output", node.format_final_output)

        # Set entry point
        graph.set_entry_point("detect_languages")

        # Sequential flow with anomaly + trending detection
        graph.add_edge("detect_languages", "vectorize_texts")
        graph.add_edge("vectorize_texts", "anomaly_detection")
        graph.add_edge("anomaly_detection", "trending_detection")
        graph.add_edge("trending_detection", "generate_expert_summary")
        graph.add_edge("generate_expert_summary", "format_output")
        graph.add_edge("format_output", END)

        return graph.compile()


# Module-level compilation
print("\n" + "=" * 60)
print("[BRAIN] BUILDING VECTORIZATION AGENT GRAPH")
print("=" * 60)
print("Architecture: 6-Step Sequential Pipeline")
print("  Step 1: Language Detection (FastText/Unicode)")
print("  Step 2: Text Vectorization (SinhalaBERTo/Tamil-BERT/DistilBERT)")
print("  Step 3: Anomaly Detection (Isolation Forest)")
print("  Step 4: Trending Detection (Velocity/Spikes)")
print("  Step 5: Expert Summary (GroqLLM)")
print("  Step 6: Format Output")
print("-" * 60)

llm = GroqLLM().get_llm()
graph = VectorizationGraphBuilder(llm).build_graph()

print("[OK] Vectorization Agent Graph compiled successfully")
print("=" * 60 + "\n")

