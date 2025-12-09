"""
src/graphs/intelligenceAgentGraph.py
MODULAR - Intelligence Agent Graph with Subgraph Architecture
Three independent modules executed in hybrid parallel/sequential pattern
"""

import uuid
from langgraph.graph import StateGraph, END
from src.states.intelligenceAgentState import IntelligenceAgentState
from src.nodes.intelligenceAgentNode import IntelligenceAgentNode
from src.llms.groqllm import GroqLLM


class IntelligenceGraphBuilder:
    """
    Builds the Intelligence Agent graph with modular subgraph architecture.

    Architecture:
    Module 1: Profile Monitoring (Twitter, Facebook, LinkedIn profiles)
    Module 2: Competitive Intelligence (Competitor mentions, Product reviews, Market intel)
    Module 3: Feed Generation (Categorize + LLM + Format)
    """

    def __init__(self, llm):
        self.llm = llm

    def build_profile_monitoring_subgraph(
        self, node: IntelligenceAgentNode
    ) -> StateGraph:
        """
        Subgraph 1: Profile Monitoring
        Monitors competitor social media profiles
        """
        subgraph = StateGraph(IntelligenceAgentState)
        subgraph.add_node("monitor_profiles", node.collect_profile_activity)
        subgraph.set_entry_point("monitor_profiles")
        subgraph.add_edge("monitor_profiles", END)

        return subgraph.compile()

    def build_competitive_intelligence_subgraph(
        self, node: IntelligenceAgentNode
    ) -> StateGraph:
        """
        Subgraph 2: Competitive Intelligence Collection
        Parallel collection of competitor mentions, product reviews, market intelligence
        """
        subgraph = StateGraph(IntelligenceAgentState)

        # Add collection nodes
        subgraph.add_node("competitor_mentions", node.collect_competitor_mentions)
        subgraph.add_node("product_reviews", node.collect_product_reviews)
        subgraph.add_node("market_intelligence", node.collect_market_intelligence)

        # Set parallel entry points
        subgraph.set_entry_point("competitor_mentions")
        subgraph.set_entry_point("product_reviews")
        subgraph.set_entry_point("market_intelligence")

        # All converge to END
        subgraph.add_edge("competitor_mentions", END)
        subgraph.add_edge("product_reviews", END)
        subgraph.add_edge("market_intelligence", END)

        return subgraph.compile()

    def build_feed_generation_subgraph(self, node: IntelligenceAgentNode) -> StateGraph:
        """
        Subgraph 3: Feed Generation
        Sequential: Categorize -> LLM Summary -> Format Output
        """
        subgraph = StateGraph(IntelligenceAgentState)

        subgraph.add_node("categorize", node.categorize_intelligence)
        subgraph.add_node("llm_summary", node.generate_llm_summary)
        subgraph.add_node("format_output", node.format_final_output)

        subgraph.set_entry_point("categorize")
        subgraph.add_edge("categorize", "llm_summary")
        subgraph.add_edge("llm_summary", "format_output")
        subgraph.add_edge("format_output", END)

        return subgraph.compile()

    def build_graph(self):
        """
        Main graph: Orchestrates 3 module subgraphs

        Flow:
        1. Module 1 (Profiles) + Module 2 (Intelligence) run in parallel
        2. Wait for both to complete
        3. Module 3 (Feed Generation) processes aggregated results
        4. Module 4 (Feed Aggregator) stores unique posts
        """
        node = IntelligenceAgentNode(self.llm)

        # Build subgraphs
        profile_subgraph = self.build_profile_monitoring_subgraph(node)
        intelligence_subgraph = self.build_competitive_intelligence_subgraph(node)
        feed_subgraph = self.build_feed_generation_subgraph(node)

        # Main graph
        main_graph = StateGraph(IntelligenceAgentState)

        # Add subgraphs as nodes
        main_graph.add_node("profile_monitoring_module", profile_subgraph.invoke)
        main_graph.add_node(
            "competitive_intelligence_module", intelligence_subgraph.invoke
        )
        main_graph.add_node("feed_generation_module", feed_subgraph.invoke)
        main_graph.add_node("feed_aggregator", node.aggregate_and_store_feeds)

        # Set parallel execution
        main_graph.set_entry_point("profile_monitoring_module")
        main_graph.set_entry_point("competitive_intelligence_module")

        # Both collection modules flow to feed generation
        main_graph.add_edge("profile_monitoring_module", "feed_generation_module")
        main_graph.add_edge("competitive_intelligence_module", "feed_generation_module")

        # Feed generation flows to aggregator
        main_graph.add_edge("feed_generation_module", "feed_aggregator")

        # Aggregator is the final step
        main_graph.add_edge("feed_aggregator", END)

        return main_graph.compile()


# Module-level compilation
print("\n" + "=" * 60)
print("🏗️  BUILDING MODULAR INTELLIGENCE AGENT GRAPH")
print("=" * 60)
print("Architecture: 3-Module Competitive Intelligence Design")
print("  Module 1: Profile Monitoring (Twitter, Facebook, LinkedIn)")
print("  Module 2: Competitive Intelligence (Mentions, Reviews, Market)")
print("  Module 3: Feed Generation (Categorize + LLM + Format)")
print("  Module 4: Feed Aggregator (Neo4j + ChromaDB + CSV)")
print("-" * 60)

llm = GroqLLM().get_llm()
graph = IntelligenceGraphBuilder(llm).build_graph()

print("✅ Intelligence Agent Graph compiled successfully")
print("=" * 60 + "\n")
