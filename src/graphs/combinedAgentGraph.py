"""
combinedAgentGraph.py
Main entry point for the Combined Agent System.
FIXED: Removed sub-graph wrappers that were causing CancelledError
"""
from __future__ import annotations
from typing import Dict, Any
import logging
from datetime import datetime

# LangGraph Imports
from langgraph.graph import StateGraph, START, END

# Project Imports
from src.llms.groqllm import GroqLLM
from src.states.combinedAgentState import CombinedAgentState
from src.nodes.combinedAgentNode import CombinedAgentNode


# Import Sub-Graph Builders
from src.graphs.socialAgentGraph import SocialGraphBuilder
from src.graphs.intelligenceAgentGraph import IntelligenceGraphBuilder
from src.graphs.economicalAgentGraph import EconomicalGraphBuilder
from src.graphs.politicalAgentGraph import PoliticalGraphBuilder
from src.graphs.meteorologicalAgentGraph import MeteorologicalGraphBuilder

# Configure Logging
logger = logging.getLogger("main_graph")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


class CombinedAgentGraphBuilder:
    def __init__(self, llm):
        self.llm = llm

    def build_graph(self):
        # 1. Initialize Sub-Graph Builders and compile them
        social_graph = SocialGraphBuilder(self.llm).build_graph()
        intelligence_graph = IntelligenceGraphBuilder(self.llm).build_graph()
        economical_graph = EconomicalGraphBuilder(self.llm).build_graph()
        political_graph = PoliticalGraphBuilder(self.llm).build_graph()
        meteorological_graph = MeteorologicalGraphBuilder(self.llm).build_graph()

        # 2. Create wrapper functions to extract domain_insights from sub-agent states
        # This solves the state type mismatch issue - sub-agents return their own state types
        # but we need to update CombinedAgentState. Wrappers extract domain_insights and
        # return update dicts that get merged via the reduce_insights reducer.
        
        def run_social_agent(state: CombinedAgentState) -> Dict[str, Any]:
            """Wrapper to invoke SocialAgent and extract domain_insights"""
            logger.info("[CombinedGraph] Invoking SocialAgent...")
            result = social_graph.invoke({})
            insights = result.get("domain_insights", [])
            logger.info(f"[CombinedGraph] SocialAgent returned {len(insights)} insights")
            return {"domain_insights": insights}
        
        def run_intelligence_agent(state: CombinedAgentState) -> Dict[str, Any]:
            """Wrapper to invoke IntelligenceAgent and extract domain_insights"""
            logger.info("[CombinedGraph] Invoking IntelligenceAgent...")
            result = intelligence_graph.invoke({})
            insights = result.get("domain_insights", [])
            logger.info(f"[CombinedGraph] IntelligenceAgent returned {len(insights)} insights")
            return {"domain_insights": insights}
        
        def run_economical_agent(state: CombinedAgentState) -> Dict[str, Any]:
            """Wrapper to invoke EconomicalAgent and extract domain_insights"""
            logger.info("[CombinedGraph] Invoking EconomicalAgent...")
            result = economical_graph.invoke({})
            insights = result.get("domain_insights", [])
            logger.info(f"[CombinedGraph] EconomicalAgent returned {len(insights)} insights")
            return {"domain_insights": insights}
        
        def run_political_agent(state: CombinedAgentState) -> Dict[str, Any]:
            """Wrapper to invoke PoliticalAgent and extract domain_insights"""
            logger.info("[CombinedGraph] Invoking PoliticalAgent...")
            result = political_graph.invoke({})
            insights = result.get("domain_insights", [])
            logger.info(f"[CombinedGraph] PoliticalAgent returned {len(insights)} insights")
            return {"domain_insights": insights}
        
        def run_meteorological_agent(state: CombinedAgentState) -> Dict[str, Any]:
            """Wrapper to invoke MeteorologicalAgent and extract domain_insights"""
            logger.info("[CombinedGraph] Invoking MeteorologicalAgent...")
            result = meteorological_graph.invoke({})
            insights = result.get("domain_insights", [])
            logger.info(f"[CombinedGraph] MeteorologicalAgent returned {len(insights)} insights")
            return {"domain_insights": insights}

        # 3. Initialize Main Orchestrator Node
        orchestrator = CombinedAgentNode(self.llm)

        # 4. Create State Graph
        workflow = StateGraph(CombinedAgentState)

        # 5. Add Sub-Agent Wrapper Nodes
        # These wrappers extract domain_insights from sub-agent results and 
        # return updates for CombinedAgentState (via the reduce_insights reducer)
        workflow.add_node("SocialAgent", run_social_agent)
        workflow.add_node("IntelligenceAgent", run_intelligence_agent)
        workflow.add_node("EconomicalAgent", run_economical_agent)
        workflow.add_node("PoliticalAgent", run_political_agent)
        workflow.add_node("MeteorologicalAgent", run_meteorological_agent)

        # 6. Add Orchestration Nodes (Fan-In)
        workflow.add_node("GraphInitiator", orchestrator.graph_initiator)
        workflow.add_node("FeedAggregatorAgent", orchestrator.feed_aggregator_agent)
        workflow.add_node("DataRefresherAgent", orchestrator.data_refresher_agent)
        workflow.add_node("DataRefreshRouter", orchestrator.data_refresh_router)

        # 7. Define Edges
        # Start -> Initiator
        workflow.add_edge(START, "GraphInitiator")

        # Initiator -> All Sub-Agents (Parallel)
        sub_agents = [
            "SocialAgent", "IntelligenceAgent", "EconomicalAgent",
            "PoliticalAgent", "MeteorologicalAgent"
        ]
        for agent in sub_agents:
            workflow.add_edge("GraphInitiator", agent)
            workflow.add_edge(agent, "FeedAggregatorAgent")

        # Aggregator -> Refresher -> Router
        workflow.add_edge("FeedAggregatorAgent", "DataRefresherAgent")
        workflow.add_edge("DataRefresherAgent", "DataRefreshRouter")

        # 8. Conditional Routing
        workflow.add_conditional_edges(
            "DataRefreshRouter",
            lambda x: x.route if x.route else "END",
            {
                "GraphInitiator": "GraphInitiator",
                "END": END
            }
        )

        return workflow.compile()

# --- GLOBAL EXPORT FOR LANGGRAPH DEV ---
# This code runs when the file is imported.
# It instantiates the LLM and builds the graph object.
print("--- BUILDING COMBINED AGENT GRAPH (FIXED: State Sync Wrappers) ---")
llm = GroqLLM().get_llm()
builder = CombinedAgentGraphBuilder(llm)
graph = builder.build_graph()
print("✓ Combined Roger Graph built successfully")
