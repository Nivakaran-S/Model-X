"""
src/graphs/socialAgentGraph.py
MODULAR - Social Agent Graph with Subgraph Architecture
Three independent modules for social intelligence collection
"""
import uuid
from langgraph.graph import StateGraph, END
from src.states.socialAgentState import SocialAgentState
from src.nodes.socialAgentNode import SocialAgentNode
from src.llms.groqllm import GroqLLM


class SocialGraphBuilder:
    """
    Builds the Social Agent graph with modular subgraph architecture.
    
    Architecture:
    Module 1: Trending Topics (Sri Lanka specific)
    Module 2: Social Media (Sri Lanka + Asia + World)
    Module 3: Feed Generation (Categorize + LLM + Format)
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def build_trending_subgraph(self, node: SocialAgentNode) -> StateGraph:
        """
        Subgraph 1: Trending Topics Collection
        Collects Sri Lankan trending topics
        """
        subgraph = StateGraph(SocialAgentState)
        subgraph.add_node("collect_trends", node.collect_sri_lanka_trends)
        subgraph.set_entry_point("collect_trends")
        subgraph.add_edge("collect_trends", END)
        
        return subgraph.compile()
    
    def build_social_media_subgraph(self, node: SocialAgentNode) -> StateGraph:
        """
        Subgraph 2: Social Media Collection
        Parallel collection across three geographic scopes
        """
        subgraph = StateGraph(SocialAgentState)
        
        # Add collection nodes
        subgraph.add_node("sri_lanka_social", node.collect_sri_lanka_social_media)
        subgraph.add_node("asia_social", node.collect_asia_social_media)
        subgraph.add_node("world_social", node.collect_world_social_media)
        
        # Set entry point (will fan out to all three)
        subgraph.set_entry_point("sri_lanka_social")
        subgraph.set_entry_point("asia_social")
        subgraph.set_entry_point("world_social")
        
        # All converge to END
        subgraph.add_edge("sri_lanka_social", END)
        subgraph.add_edge("asia_social", END)
        subgraph.add_edge("world_social", END)
        
        return subgraph.compile()
    
    def build_feed_generation_subgraph(self, node: SocialAgentNode) -> StateGraph:
        """
        Subgraph 3: Feed Generation
        Sequential: Categorize → LLM Summary → Format Output
        """
        subgraph = StateGraph(SocialAgentState)
        
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
        1. Module 1 (Trending) + Module 2 (Social) run in parallel
        2. Wait for both to complete
        3. Module 3 (Feed Generation) processes aggregated results
        4. Module 4 (Feed Aggregator) stores unique posts
        """
        node = SocialAgentNode(self.llm)
        
        # Build subgraphs
        trending_subgraph = self.build_trending_subgraph(node)
        social_subgraph = self.build_social_media_subgraph(node)
        feed_subgraph = self.build_feed_generation_subgraph(node)
        
        # Main graph
        main_graph = StateGraph(SocialAgentState)
        
        # Add subgraphs as nodes
        main_graph.add_node("trending_module", trending_subgraph.invoke)
        main_graph.add_node("social_media_module", social_subgraph.invoke)
        main_graph.add_node("feed_generation_module", feed_subgraph.invoke)
        main_graph.add_node("feed_aggregator", node.aggregate_and_store_feeds)
        
        # Set parallel execution
        main_graph.set_entry_point("trending_module")
        main_graph.set_entry_point("social_media_module")
        
        # Both collection modules flow to feed generation
        main_graph.add_edge("trending_module", "feed_generation_module")
        main_graph.add_edge("social_media_module", "feed_generation_module")
        
        # Feed generation flows to aggregator
        main_graph.add_edge("feed_generation_module", "feed_aggregator")
        
        # Aggregator is the final step
        main_graph.add_edge("feed_aggregator", END)
        
        return main_graph.compile()


# Module-level compilation
print("\n" + "="*60)
print("[BUILD] MODULAR SOCIAL AGENT GRAPH")
print("="*60)
print("Architecture: 3-Module Hybrid Design")
print("  Module 1: Trending Topics (Sri Lanka specific)")
print("  Module 2: Social Media (5 platforms × 3 geographic scopes)")
print("  Module 3: Feed Generation (Categorize + LLM + Format)")
print("  Module 4: Feed Aggregator (Neo4j + ChromaDB + CSV)")
print("-"*60)

llm = GroqLLM().get_llm()
graph = SocialGraphBuilder(llm).build_graph()

print("[OK] Social Agent Graph compiled successfully")
print("="*60 + "\n")
