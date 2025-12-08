"""
src/graphs/RogerGraph.py
COMPLETE - Main Roger Graph with Fan-Out/Fan-In Architecture
This is the "Mother Graph" that orchestrates all domain agents
"""
from __future__ import annotations
import logging
from langgraph.graph import StateGraph, START, END

# State and Node imports
from src.states.combinedAgentState import CombinedAgentState
from src.nodes.combinedAgentNode import CombinedAgentNode

# Domain graph builders
from src.graphs.dataRetrievalAgentGraph import DataRetrievalAgentGraph
from src.graphs.meteorologicalAgentGraph import MeteorologicalGraphBuilder
from src.graphs.politicalAgentGraph import PoliticalGraphBuilder
from src.graphs.economicalAgentGraph import EconomicalGraphBuilder
from src.graphs.intelligenceAgentGraph import IntelligenceGraphBuilder
from src.graphs.socialAgentGraph import SocialGraphBuilder

from src.llms.groqllm import GroqLLM

logger = logging.getLogger("Roger_graph")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


class CombinedAgentGraphBuilder:
    """
    Builds the main Roger graph implementing Fan-Out/Fan-In architecture.
    
    Architecture:
    1. GraphInitiator (START)
    2. Fan-Out to 6 Domain Agents (parallel execution)
    3. Fan-In to FeedAggregator (collects domain_insights)
    4. DataRefresher (updates dashboard)
    5. DataRefreshRouter (loop or end decision)
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def build_graph(self):
        logger.info("=" * 60)
        logger.info("BUILDING Roger COMBINED AGENT GRAPH")
        logger.info("=" * 60)
        
        # 1. Instantiate domain graph builders
        social_builder = SocialGraphBuilder(self.llm)
        intelligence_builder = IntelligenceGraphBuilder(self.llm)
        economical_builder = EconomicalGraphBuilder(self.llm)
        political_builder = PoliticalGraphBuilder(self.llm)
        meteorological_builder = MeteorologicalGraphBuilder(self.llm)
        data_retrieval_builder = DataRetrievalAgentGraph(self.llm)
        
        logger.info("✓ Domain graph builders instantiated")
        
        # 2. Instantiate orchestration node
        orchestrator = CombinedAgentNode(self.llm)
        logger.info("✓ Orchestration node instantiated")
        
        # 3. Create state graph with CombinedAgentState
        workflow = StateGraph(CombinedAgentState)
        logger.info("✓ StateGraph created with CombinedAgentState")
        
        # 4. Add orchestration nodes
        workflow.add_node("GraphInitiator", orchestrator.graph_initiator)
        workflow.add_node("FeedAggregatorAgent", orchestrator.feed_aggregator_agent)
        workflow.add_node("DataRefresherAgent", orchestrator.data_refresher_agent)
        workflow.add_node("DataRefreshRouter", orchestrator.data_refresh_router)
        logger.info("✓ Orchestration nodes added")
        
        # 5. Add domain subgraphs (compiled graphs as nodes)
        workflow.add_node("SocialAgent", social_builder.build_graph())
        workflow.add_node("IntelligenceAgent", intelligence_builder.build_graph())
        workflow.add_node("EconomicalAgent", economical_builder.build_graph())
        workflow.add_node("PoliticalAgent", political_builder.build_graph())
        workflow.add_node("MeteorologicalAgent", meteorological_builder.build_graph())
        workflow.add_node("DataRetrievalAgent", data_retrieval_builder.build_data_retrieval_agent_graph())
        logger.info("✓ Domain agent subgraphs added")
        
        # 6. Wire the graph: START -> Initiator
        workflow.add_edge(START, "GraphInitiator")
        
        # 7. Fan-Out: Initiator -> All Domain Agents (parallel execution)
        domain_agents = [
            "SocialAgent",
            "IntelligenceAgent",
            "EconomicalAgent",
            "PoliticalAgent",
            "MeteorologicalAgent",
            "DataRetrievalAgent"
        ]
        
        for agent in domain_agents:
            workflow.add_edge("GraphInitiator", agent)
        
        logger.info(f"✓ Fan-Out configured: GraphInitiator -> {len(domain_agents)} agents")
        
        # 8. Fan-In: All Domain Agents -> FeedAggregator
        for agent in domain_agents:
            workflow.add_edge(agent, "FeedAggregatorAgent")
        
        logger.info(f"✓ Fan-In configured: {len(domain_agents)} agents -> FeedAggregator")
        
        # 9. Linear flow: Aggregator -> Refresher -> Router
        workflow.add_edge("FeedAggregatorAgent", "DataRefresherAgent")
        workflow.add_edge("DataRefresherAgent", "DataRefreshRouter")
        logger.info("✓ Linear orchestration flow configured")
        
        # 10. Conditional routing: Router -> Loop or END
        def route_decision(state):
            """
            Router function for conditional edges.
            Returns the next node name or END.
            """
            route = getattr(state, "route", [])
            
            # If route is None or empty, go to END
            if route is None or route == "":
                return END
            
            # If route is "GraphInitiator", loop back
            if route == "GraphInitiator":
                return "GraphInitiator"
            
            # Default to END
            return END
        
        workflow.add_conditional_edges(
            "DataRefreshRouter",
            route_decision,
            {
                "GraphInitiator": "GraphInitiator",
                END: END
            }
        )
        logger.info("✓ Conditional routing configured")
        
        # 11. Compile the graph
        graph = workflow.compile()
        
        logger.info("=" * 60)
        logger.info("✓ Roger GRAPH COMPILED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Graph Structure:")
        logger.info("  START")
        logger.info("    ↓")
        logger.info("  GraphInitiator")
        logger.info("    ↓↓↓↓↓↓ (Fan-Out)")
        logger.info("  [Social, Intelligence, Economic, Political, Meteorological, DataRetrieval]")
        logger.info("    ↓↓↓↓↓↓ (Fan-In)")
        logger.info("  FeedAggregatorAgent")
        logger.info("    ↓")
        logger.info("  DataRefresherAgent")
        logger.info("    ↓")
        logger.info("  DataRefreshRouter")
        logger.info("    ↓ (conditional)")
        logger.info("  [GraphInitiator (loop) OR END]")
        logger.info("")
        
        return graph


# Module-level compilation for LangGraph CLI
print("\n" + "=" * 60)
print("INITIALIZING Roger PLATFORM")
print("=" * 60)
llm = GroqLLM().get_llm()
builder = CombinedAgentGraphBuilder(llm)
graph = builder.build_graph()
print("\n✓ Roger Platform Ready")
print("=" * 60)
