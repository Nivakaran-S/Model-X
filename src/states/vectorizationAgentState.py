"""
src/states/vectorizationAgentState.py
Vectorization Agent State - handles text-to-vector conversion with multilingual BERT
"""

from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict


class VectorizationAgentState(TypedDict, total=False):
    """
    State for Vectorization Agent.
    Converts text to vectors using language-specific BERT models.
    Steps: Language Detection → Vectorization → Expert Summary

    Note: This is a sequential graph, so no reducers needed.
    Each node's output fully replaces the field value.
    """

    # ===== INPUT =====
    input_texts: List[Dict[str, Any]]  # [{text, post_id, metadata}]
    batch_id: str

    # ===== LANGUAGE DETECTION =====
    language_detection_results: List[Dict[str, Any]]
    # [{post_id, text, language, confidence}]

    # ===== VECTORIZATION =====
    vector_embeddings: List[Dict[str, Any]]
    # [{post_id, language, vector, model_used}]

    # ===== CLUSTERING/ANOMALY =====
    clustering_results: Optional[Dict[str, Any]]
    anomaly_results: Optional[Dict[str, Any]]

    # ===== EXPERT ANALYSIS =====
    expert_summary: Optional[str]  # LLM-generated summary combining all insights
    opportunities: List[Dict[str, Any]]  # Detected opportunities
    threats: List[Dict[str, Any]]  # Detected threats

    # ===== PROCESSING STATUS =====
    current_step: str
    processing_stats: Dict[str, Any]
    errors: List[str]

    # ===== LLM OUTPUT =====
    llm_response: Optional[str]
    structured_output: Dict[str, Any]

    # ===== INTEGRATION WITH PARENT GRAPH =====
    domain_insights: List[Dict[str, Any]]

    # ===== FINAL OUTPUT =====
    final_output: Dict[str, Any]
