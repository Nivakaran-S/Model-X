"""
vectorizationAgentGraph.py - Vectorization Agent Graph for text-to-vector conversion
"""

from langgraph.graph import StateGraph, END
from src.states.vectorizationAgentState import VectorizationAgentState
from src.nodes.vectorizationAgentNode import VectorizationAgentNode
from src.llms.groqllm import GroqLLM


class VectorizationGraphBuilder:
    def __init__(self, llm=None):
        self.llm = llm or GroqLLM().get_llm()

    def build_graph(self):
        node = VectorizationAgentNode(self.llm)

        graph = StateGraph(VectorizationAgentState)

        graph.add_node("detect_languages", node.detect_languages)
        graph.add_node("vectorize_texts", node.vectorize_texts)
        graph.add_node("anomaly_detection", node.run_anomaly_detection)
        graph.add_node("trending_detection", node.run_trending_detection)
        graph.add_node("generate_expert_summary", node.generate_expert_summary)
        graph.add_node("format_output", node.format_final_output)

        graph.set_entry_point("detect_languages")

        graph.add_edge("detect_languages", "vectorize_texts")
        graph.add_edge("vectorize_texts", "anomaly_detection")
        graph.add_edge("anomaly_detection", "trending_detection")
        graph.add_edge("trending_detection", "generate_expert_summary")
        graph.add_edge("generate_expert_summary", "format_output")
        graph.add_edge("format_output", END)

        return graph.compile()


llm = GroqLLM().get_llm()
graph = VectorizationGraphBuilder(llm).build_graph()
