"""
src/graphs/meteorologicalAgentGraph.py
MODULAR - Meteorological Agent Graph with Subgraph Architecture
Three independent modules executed in parallel
"""

import uuid
from langgraph.graph import StateGraph, END
from src.states.meteorologicalAgentState import MeteorologicalAgentState
from src.nodes.meteorologicalAgentNode import MeteorologicalAgentNode
from src.llms.groqllm import GroqLLM


class MeteorologicalGraphBuilder:
    """
    Builds the Meteorological Agent graph with modular subgraph architecture.

    Architecture:
    Module 1: Official Weather Sources (DMC + Weather Nowcast)
    Module 2: Social Media (National + Districts + Climate)
    Module 3: Feed Generation (Categorize + LLM + Format)
    """

    def __init__(self, llm):
        self.llm = llm

    def build_official_sources_subgraph(
        self, node: MeteorologicalAgentNode
    ) -> StateGraph:
        """
        Subgraph 1: Official Weather Sources Collection
        Collects DMC alerts and weather nowcast data
        """
        subgraph = StateGraph(MeteorologicalAgentState)
        subgraph.add_node("collect_official", node.collect_official_sources)
        subgraph.set_entry_point("collect_official")
        subgraph.add_edge("collect_official", END)

        return subgraph.compile()

    def build_social_media_subgraph(self, node: MeteorologicalAgentNode) -> StateGraph:
        """
        Subgraph 2: Social Media Collection
        Parallel collection of national, district, and climate weather media
        """
        subgraph = StateGraph(MeteorologicalAgentState)

        # Add collection nodes
        subgraph.add_node("national_social", node.collect_national_social_media)
        subgraph.add_node("district_social", node.collect_district_social_media)
        subgraph.add_node("climate_alerts", node.collect_climate_alerts)

        # Set entry point (will fan out to all three)
        subgraph.set_entry_point("national_social")
        subgraph.set_entry_point("district_social")
        subgraph.set_entry_point("climate_alerts")

        # All converge to END
        subgraph.add_edge("national_social", END)
        subgraph.add_edge("district_social", END)
        subgraph.add_edge("climate_alerts", END)

        return subgraph.compile()

    def build_feed_generation_subgraph(
        self, node: MeteorologicalAgentNode
    ) -> StateGraph:
        """
        Subgraph 3: Feed Generation
        Sequential: Categorize → LLM Summary → Format Output
        """
        subgraph = StateGraph(MeteorologicalAgentState)

        subgraph.add_node("categorize", node.categorize_by_geography)
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
        1. Module 1 (Official) + Module 2 (Social) run in parallel
        2. Wait for both to complete
        3. Module 3 (Feed Generation) processes aggregated results
        4. Module 4 (Feed Aggregator) stores unique posts
        """
        node = MeteorologicalAgentNode(self.llm)

        # Build subgraphs
        official_subgraph = self.build_official_sources_subgraph(node)
        social_subgraph = self.build_social_media_subgraph(node)
        feed_subgraph = self.build_feed_generation_subgraph(node)

        # Main graph
        main_graph = StateGraph(MeteorologicalAgentState)

        # Add subgraphs as nodes
        main_graph.add_node("official_sources_module", official_subgraph.invoke)
        main_graph.add_node("social_media_module", social_subgraph.invoke)
        main_graph.add_node("feed_generation_module", feed_subgraph.invoke)
        main_graph.add_node("feed_aggregator", node.aggregate_and_store_feeds)

        # Set parallel execution
        main_graph.set_entry_point("official_sources_module")
        main_graph.set_entry_point("social_media_module")

        # Both collection modules flow to feed generation
        main_graph.add_edge("official_sources_module", "feed_generation_module")
        main_graph.add_edge("social_media_module", "feed_generation_module")

        # Feed generation flows to aggregator
        main_graph.add_edge("feed_generation_module", "feed_aggregator")

        # Aggregator is the final step
        main_graph.add_edge("feed_aggregator", END)

        return main_graph.compile()


# Module-level compilation
print("\n" + "=" * 60)
print("🏗️  BUILDING MODULAR METEOROLOGICAL AGENT GRAPH")
print("=" * 60)
print("Architecture: 3-Module Hybrid Design")
print("  Module 1: Official Sources (DMC Alerts + Weather Nowcast)")
print("  Module 2: Social Media (5 platforms × 3 scopes)")
print("  Module 3: Feed Generation (Categorize + LLM + Format)")
print("  Module 4: Feed Aggregator (Neo4j + ChromaDB + CSV)")
print("-" * 60)

llm = GroqLLM().get_llm()
graph = MeteorologicalGraphBuilder(llm).build_graph()

print("✅ Meteorological Agent Graph compiled successfully")
print("=" * 60 + "\n")
