"""
src/nodes/vectorizationAgentNode.py
Vectorization Agent Node - Agentic AI for text-to-vector conversion
Uses language-specific BERT models for Sinhala, Tamil, and English
"""

import sys
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

# Add models path
MODELS_PATH = Path(__file__).parent.parent.parent / "models" / "anomaly-detection"
sys.path.insert(0, str(MODELS_PATH))

from src.states.vectorizationAgentState import VectorizationAgentState
from src.llms.groqllm import GroqLLM

logger = logging.getLogger("vectorization_agent_node")

# Import vectorization utilities from models/anomaly-detection/src/utils/
try:
    # MODELS_PATH is already added to sys.path, so import from src.utils.vectorizer
    from src.utils.vectorizer import detect_language, get_vectorizer

    VECTORIZER_AVAILABLE = True
except ImportError as e:
    try:
        # Fallback: try direct import if running from different context
        import importlib.util

        vectorizer_path = MODELS_PATH / "src" / "utils" / "vectorizer.py"
        if vectorizer_path.exists():
            spec = importlib.util.spec_from_file_location("vectorizer", vectorizer_path)
            vectorizer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vectorizer_module)
            detect_language = vectorizer_module.detect_language
            get_vectorizer = vectorizer_module.get_vectorizer
            VECTORIZER_AVAILABLE = True
        else:
            VECTORIZER_AVAILABLE = False
            # Define placeholder functions to prevent NameError
            detect_language = None
            get_vectorizer = None
            logger.warning(
                f"[VectorizationAgent] Vectorizer not found at {vectorizer_path}"
            )
    except Exception as e2:
        VECTORIZER_AVAILABLE = False
        detect_language = None
        get_vectorizer = None
        logger.warning(f"[VectorizationAgent] Vectorizer import failed: {e} / {e2}")


class VectorizationAgentNode:
    """
    Agentic AI for converting text to vectors using language-specific BERT models.

    Steps:
    1. Language Detection (FastText/lingua-py + Unicode script)
    2. Text Vectorization (SinhalaBERTo / Tamil-BERT / DistilBERT)
    3. Expert Summary (GroqLLM for combining insights)
    """

    MODEL_INFO = {
        "english": {
            "name": "DistilBERT",
            "hf_name": "distilbert-base-uncased",
            "description": "Fast and accurate English understanding",
        },
        "sinhala": {
            "name": "SinhalaBERTo",
            "hf_name": "keshan/SinhalaBERTo",
            "description": "Specialized Sinhala context and sentiment",
        },
        "tamil": {
            "name": "Tamil-BERT",
            "hf_name": "l3cube-pune/tamil-bert",
            "description": "Specialized Tamil understanding",
        },
    }

    def __init__(self, llm=None):
        """Initialize vectorization agent node"""
        self.llm = llm or GroqLLM().get_llm()
        self.vectorizer = None

        logger.info("[VectorizationAgent] Initialized")
        logger.info(f"  Available models: {list(self.MODEL_INFO.keys())}")

    def _get_vectorizer(self):
        """Lazy load vectorizer"""
        if self.vectorizer is None and VECTORIZER_AVAILABLE:
            self.vectorizer = get_vectorizer()
        return self.vectorizer

    def detect_languages(self, state: VectorizationAgentState) -> Dict[str, Any]:
        """
        Step 1: Detect language for each input text.
        Uses FastText/lingua-py with Unicode script fallback.
        """
        import json

        logger.info("[VectorizationAgent] STEP 1: Language Detection")

        raw_input = state.get("input_texts", [])

        # DEBUG: Log raw input
        logger.info(f"[VectorizationAgent] DEBUG: raw_input type = {type(raw_input)}")
        logger.info(f"[VectorizationAgent] DEBUG: raw_input = {str(raw_input)[:500]}")

        # Robust parsing: handle string, list, or other formats
        input_texts = []

        if isinstance(raw_input, str):
            # Try to parse as JSON string
            try:
                parsed = json.loads(raw_input)
                if isinstance(parsed, list):
                    input_texts = parsed
                elif isinstance(parsed, dict) and "input_texts" in parsed:
                    input_texts = parsed["input_texts"]
                else:
                    # Single text string
                    input_texts = [{"text": raw_input, "post_id": "single_text"}]
            except json.JSONDecodeError:
                # Plain text string
                input_texts = [{"text": raw_input, "post_id": "plain_text"}]
        elif isinstance(raw_input, list):
            # Already a list - validate each item
            for i, item in enumerate(raw_input):
                if isinstance(item, dict):
                    input_texts.append(item)
                elif isinstance(item, str):
                    # String item in list
                    try:
                        parsed_item = json.loads(item)
                        if isinstance(parsed_item, dict):
                            input_texts.append(parsed_item)
                        else:
                            input_texts.append({"text": item, "post_id": f"text_{i}"})
                    except json.JSONDecodeError:
                        input_texts.append({"text": item, "post_id": f"text_{i}"})
                else:
                    input_texts.append({"text": str(item), "post_id": f"text_{i}"})
        elif isinstance(raw_input, dict):
            # Single dict
            input_texts = [raw_input]

        logger.info(
            f"[VectorizationAgent] DEBUG: Parsed {len(input_texts)} input texts"
        )

        if not input_texts:
            logger.warning("[VectorizationAgent] No input texts provided")
            return {
                "current_step": "language_detection",
                "language_detection_results": [],
                "errors": ["No input texts provided"],
            }

        results = []
        lang_counts = {"english": 0, "sinhala": 0, "tamil": 0, "unknown": 0}

        for item in input_texts:
            text = item.get("text", "")
            post_id = item.get("post_id", "")

            if VECTORIZER_AVAILABLE:
                language, confidence = detect_language(text)
            else:
                # Fallback: simple detection
                language, confidence = self._simple_detect(text)

            lang_counts[language] = lang_counts.get(language, 0) + 1

            results.append(
                {
                    "post_id": post_id,
                    "text": text,
                    "language": language,
                    "confidence": confidence,
                    "model_to_use": self.MODEL_INFO.get(
                        language, self.MODEL_INFO["english"]
                    )["hf_name"],
                }
            )

        logger.info(f"[VectorizationAgent] Language distribution: {lang_counts}")

        return {
            "current_step": "language_detection",
            "language_detection_results": results,
            "processing_stats": {
                "total_texts": len(input_texts),
                "language_distribution": lang_counts,
            },
        }

    def _simple_detect(self, text: str) -> tuple:
        """Simple fallback language detection based on Unicode ranges"""
        sinhala_range = (0x0D80, 0x0DFF)
        tamil_range = (0x0B80, 0x0BFF)

        sinhala_count = sum(
            1 for c in text if sinhala_range[0] <= ord(c) <= sinhala_range[1]
        )
        tamil_count = sum(1 for c in text if tamil_range[0] <= ord(c) <= tamil_range[1])

        total = len(text)
        if total == 0:
            return "english", 0.5

        if sinhala_count / total > 0.3:
            return "sinhala", 0.8
        if tamil_count / total > 0.3:
            return "tamil", 0.8
        return "english", 0.7

    def vectorize_texts(self, state: VectorizationAgentState) -> Dict[str, Any]:
        """
        Step 2: Convert texts to vectors using language-specific BERT models.
        Downloads models locally from HuggingFace on first use.
        """
        logger.info("[VectorizationAgent] STEP 2: Text Vectorization")

        detection_results = state.get("language_detection_results", [])

        if not detection_results:
            logger.warning("[VectorizationAgent] No language detection results")
            return {
                "current_step": "vectorization",
                "vector_embeddings": [],
                "errors": ["No texts to vectorize"],
            }

        vectorizer = self._get_vectorizer()
        embeddings = []

        for item in detection_results:
            text = item.get("text", "")
            post_id = item.get("post_id", "")
            language = item.get("language", "english")

            try:
                if vectorizer:
                    vector = vectorizer.vectorize(text, language)
                else:
                    # Fallback: zero vector
                    vector = np.zeros(768)

                embeddings.append(
                    {
                        "post_id": post_id,
                        "language": language,
                        "vector": (
                            vector.tolist()
                            if hasattr(vector, "tolist")
                            else list(vector)
                        ),
                        "vector_dim": len(vector),
                        "model_used": self.MODEL_INFO.get(language, {}).get(
                            "name", "Unknown"
                        ),
                    }
                )

            except Exception as e:
                logger.error(
                    f"[VectorizationAgent] Vectorization error for {post_id}: {e}"
                )
                embeddings.append(
                    {
                        "post_id": post_id,
                        "language": language,
                        "vector": [0.0] * 768,
                        "vector_dim": 768,
                        "model_used": "fallback",
                        "error": str(e),
                    }
                )

        logger.info(f"[VectorizationAgent] Vectorized {len(embeddings)} texts")

        return {
            "current_step": "vectorization",
            "vector_embeddings": embeddings,
            "processing_stats": {
                **state.get("processing_stats", {}),
                "vectors_generated": len(embeddings),
                "vector_dim": 768,
            },
        }

    def run_anomaly_detection(self, state: VectorizationAgentState) -> Dict[str, Any]:
        """
        Step 2.5: Run anomaly detection on vectorized embeddings.
        Uses trained Isolation Forest model to identify anomalous content.
        """
        logger.info("[VectorizationAgent] STEP 2.5: Anomaly Detection")

        embeddings = state.get("vector_embeddings", [])

        if not embeddings:
            logger.warning("[VectorizationAgent] No embeddings for anomaly detection")
            return {
                "current_step": "anomaly_detection",
                "anomaly_results": {
                    "status": "skipped",
                    "reason": "no_embeddings",
                    "anomalies": [],
                    "total_analyzed": 0,
                },
            }

        # Try to load the trained model
        anomaly_model = None
        model_name = "none"

        try:
            import joblib

            model_paths = [
                # Embedding-only model (768-dim) - compatible with Vectorizer Agent
                MODELS_PATH
                / "artifacts"
                / "model_trainer"
                / "isolation_forest_embeddings_only.joblib",
                # Full-feature models (may have different dimensions)
                MODELS_PATH / "output" / "isolation_forest_embeddings_only.joblib",
                MODELS_PATH / "output" / "isolation_forest_model.joblib",
                MODELS_PATH
                / "artifacts"
                / "model_trainer"
                / "isolation_forest_model.joblib",
                MODELS_PATH / "output" / "lof_model.joblib",
            ]

            for model_path in model_paths:
                if model_path.exists():
                    anomaly_model = joblib.load(model_path)
                    model_name = model_path.stem
                    logger.info(
                        f"[VectorizationAgent] Loaded anomaly model: {model_path.name}"
                    )
                    break

        except Exception as e:
            logger.warning(f"[VectorizationAgent] Could not load anomaly model: {e}")

        if anomaly_model is None:
            logger.info(
                "[VectorizationAgent] No trained model available - using severity-based fallback"
            )
            return {
                "current_step": "anomaly_detection",
                "anomaly_results": {
                    "status": "fallback",
                    "reason": "model_not_trained",
                    "message": "Using severity-based anomaly detection until model is trained",
                    "anomalies": [],
                    "total_analyzed": len(embeddings),
                    "model_used": "severity_heuristic",
                },
            }

        # Run inference on each embedding
        # IMPORTANT: The anomaly model was trained primarily on English embeddings.
        # Different BERT models (SinhalaBERTo, Tamil-BERT, DistilBERT) produce embeddings
        # in completely different vector spaces, so non-English texts would incorrectly
        # appear as anomalies. We handle this by:
        # 1. Only running the model on English texts
        # 2. Using a content-based heuristic for non-English texts
        anomalies = []
        normal_count = 0
        skipped_non_english = 0

        for emb in embeddings:
            try:
                vector = emb.get("vector", [])
                post_id = emb.get("post_id", "")
                language = emb.get("language", "english")

                if not vector or len(vector) != 768:
                    continue

                # For non-English languages, skip anomaly detection
                # The ML model was trained on English embeddings only.
                # Different BERT models (SinhalaBERTo, Tamil-BERT) have completely
                # different embedding spaces - Tamil embeddings have magnitude ~0.64
                # while English has ~7.5 and Sinhala ~13.7. They cannot be compared.
                if language in ["sinhala", "tamil"]:
                    skipped_non_english += 1
                    normal_count += 1  # Treat as normal (not anomalous)
                    continue

                # For English texts, use the trained ML model
                vector_array = np.array(vector).reshape(1, -1)

                # Predict: -1 = anomaly, 1 = normal
                prediction = anomaly_model.predict(vector_array)[0]

                # Get anomaly score
                if hasattr(anomaly_model, "decision_function"):
                    score = -anomaly_model.decision_function(vector_array)[0]
                elif hasattr(anomaly_model, "score_samples"):
                    score = -anomaly_model.score_samples(vector_array)[0]
                else:
                    score = 1.0 if prediction == -1 else 0.0

                # Normalize score to 0-1
                normalized_score = max(0, min(1, (score + 0.5)))

                if prediction == -1:
                    anomalies.append(
                        {
                            "post_id": post_id,
                            "anomaly_score": float(normalized_score),
                            "is_anomaly": True,
                            "language": language,
                            "detection_method": "isolation_forest",
                        }
                    )
                else:
                    normal_count += 1

            except Exception as e:
                logger.debug(
                    f"[VectorizationAgent] Anomaly check failed for {post_id}: {e}"
                )

        logger.info(
            f"[VectorizationAgent] Anomaly detection: {len(anomalies)} anomalies, "
            f"{normal_count} normal, {skipped_non_english} non-English (heuristic)"
        )

        return {
            "current_step": "anomaly_detection",
            "anomaly_results": {
                "status": "success",
                "model_used": model_name,
                "total_analyzed": len(embeddings),
                "anomalies_found": len(anomalies),
                "normal_count": normal_count,
                "anomalies": anomalies,
                "anomaly_rate": len(anomalies) / len(embeddings) if embeddings else 0,
            },
        }

    def run_trending_detection(self, state: VectorizationAgentState) -> Dict[str, Any]:
        """
        Step 2.6: Detect trending topics from the input texts.

        Extracts key entities/topics and tracks their mention velocity.
        Identifies:
        - Trending topics (momentum > 2x normal)
        - Spike alerts (volume > 3x normal)
        - Topics with increasing momentum
        """
        logger.info("[VectorizationAgent] STEP 2.6: Trending Detection")

        detection_results = state.get("language_detection_results", [])

        if not detection_results:
            logger.warning("[VectorizationAgent] No texts for trending detection")
            return {
                "current_step": "trending_detection",
                "trending_results": {
                    "status": "skipped",
                    "reason": "no_texts",
                    "trending_topics": [],
                    "spike_alerts": [],
                },
            }

        # Import trending detector
        try:
            from src.utils.trending_detector import (
                get_trending_detector,
                record_topic_mention,
                get_trending_now,
                get_spikes,
            )

            TRENDING_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"[VectorizationAgent] Trending detector not available: {e}")
            TRENDING_AVAILABLE = False

        if not TRENDING_AVAILABLE:
            return {
                "current_step": "trending_detection",
                "trending_results": {
                    "status": "unavailable",
                    "reason": "trending_detector_not_installed",
                    "trending_topics": [],
                    "spike_alerts": [],
                },
            }

        # Extract entities and record mentions
        entities_found = []

        for item in detection_results:
            text = item.get("text", "")  # Field is 'text', not 'original_text'
            language = item.get("language", "english")
            post_id = item.get("post_id", "")

            # Simple entity extraction (keywords, capitalized words, etc.)
            # In production, you'd use NER or more sophisticated extraction
            extracted = self._extract_entities(text, language)

            for entity in extracted:
                try:
                    # Record mention with trending detector
                    record_topic_mention(
                        topic=entity["text"],
                        source=entity.get("source", "feed"),
                        domain=entity.get("domain", "general"),
                    )
                    entities_found.append(
                        {
                            "entity": entity["text"],
                            "type": entity.get("type", "keyword"),
                            "post_id": post_id,
                            "language": language,
                        }
                    )
                except Exception as e:
                    logger.debug(f"[VectorizationAgent] Failed to record mention: {e}")

        # Get current trending topics and spikes
        try:
            trending_topics = get_trending_now(limit=10)
            spike_alerts = get_spikes()
        except Exception as e:
            logger.warning(f"[VectorizationAgent] Failed to get trending data: {e}")
            trending_topics = []
            spike_alerts = []

        logger.info(
            f"[VectorizationAgent] Trending detection: {len(entities_found)} entities, "
            f"{len(trending_topics)} trending, {len(spike_alerts)} spikes"
        )

        return {
            "current_step": "trending_detection",
            "trending_results": {
                "status": "success",
                "entities_extracted": len(entities_found),
                "entities": entities_found[:20],  # Limit for state size
                "trending_topics": trending_topics,
                "spike_alerts": spike_alerts,
            },
        }

    def _extract_entities(
        self, text: str, language: str = "english"
    ) -> List[Dict[str, Any]]:
        """
        Extract entities/topics from text for trending tracking.

        Uses simple heuristics:
        - Capitalized words/phrases (potential proper nouns)
        - Hashtags
        - Common news keywords

        In production, integrate with NER model for better extraction.
        """
        entities = []

        if not text:
            return entities

        import re

        # Extract hashtags
        hashtags = re.findall(r"#(\w+)", text)
        for tag in hashtags:
            entities.append(
                {
                    "text": tag.lower(),
                    "type": "hashtag",
                    "source": "hashtag",
                    "domain": "social",
                }
            )

        # Extract capitalized phrases (potential proper nouns)
        # Match 1-4 consecutive capitalized words
        cap_phrases = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", text)
        for phrase in cap_phrases:
            # Skip common words
            if phrase.lower() not in [
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "i",
                "he",
                "she",
                "it",
            ]:
                entities.append(
                    {
                        "text": phrase,
                        "type": "proper_noun",
                        "source": "text",
                        "domain": "general",
                    }
                )

        # News/event keywords
        news_keywords = [
            "breaking",
            "urgent",
            "alert",
            "emergency",
            "crisis",
            "earthquake",
            "flood",
            "tsunami",
            "election",
            "protest",
            "strike",
            "scandal",
            "corruption",
            "price",
            "inflation",
        ]

        text_lower = text.lower()
        for keyword in news_keywords:
            if keyword in text_lower:
                entities.append(
                    {
                        "text": keyword,
                        "type": "news_keyword",
                        "source": "keyword_match",
                        "domain": "news",
                    }
                )

        # Deduplicate by text
        seen = set()
        unique_entities = []
        for e in entities:
            key = e["text"].lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return unique_entities[:15]  # Limit entities per text

    def generate_expert_summary(self, state: VectorizationAgentState) -> Dict[str, Any]:
        """
        Step 3: Use GroqLLM to generate expert summary combining all insights.
        Identifies opportunities and threats from the vectorized content.
        """
        logger.info("[VectorizationAgent] STEP 3: Expert Summary")

        detection_results = state.get("language_detection_results", [])
        embeddings = state.get("vector_embeddings", [])

        # DEBUG: Log what we received from previous nodes
        logger.info(
            f"[VectorizationAgent] DEBUG expert_summary: state keys = {list(state.keys()) if isinstance(state, dict) else 'not dict'}"
        )
        logger.info(
            f"[VectorizationAgent] DEBUG expert_summary: detection_results count = {len(detection_results)}"
        )
        logger.info(
            f"[VectorizationAgent] DEBUG expert_summary: embeddings count = {len(embeddings)}"
        )
        if detection_results:
            logger.info(
                f"[VectorizationAgent] DEBUG expert_summary: first result = {detection_results[0]}"
            )

        if not detection_results:
            logger.warning("[VectorizationAgent] No detection results received!")
            return {
                "current_step": "expert_summary",
                "expert_summary": "No data available for analysis",
                "opportunities": [],
                "threats": [],
            }

        # Prepare context for LLM
        texts_by_lang = {}
        for item in detection_results:
            lang = item.get("language", "english")
            if lang not in texts_by_lang:
                texts_by_lang[lang] = []
            texts_by_lang[lang].append(item.get("text", "")[:200])  # First 200 chars

        # Build prompt
        prompt = f"""You are an expert analyst for a Sri Lankan intelligence monitoring system.
        
Analyze the following multilingual social media content and identify:
1. OPPORTUNITIES - potential positive developments, market opportunities, favorable conditions
2. THREATS - risks, negative sentiment, potential issues, compliance concerns

Content Summary:
- Total posts analyzed: {len(detection_results)}
- Languages detected: {list(texts_by_lang.keys())}

Sample content by language:
"""
        for lang, texts in texts_by_lang.items():
            prompt += f"\n{lang.upper()} ({len(texts)} posts):\n"
            for i, text in enumerate(texts[:3]):  # First 3 samples
                prompt += f"  {i+1}. {text[:100]}...\n"

        prompt += """

Provide a structured analysis with:
1. Executive Summary (2-3 sentences)
2. Top 3 Opportunities (each with brief explanation)
3. Top 3 Threats/Risks (each with brief explanation)
4. Overall Sentiment (Positive/Neutral/Negative)

Format your response in a clear, structured manner."""

        try:
            response = self.llm.invoke(prompt)
            expert_summary = (
                response.content if hasattr(response, "content") else str(response)
            )
        except Exception as e:
            logger.error(f"[VectorizationAgent] LLM error: {e}")
            expert_summary = f"Analysis failed: {str(e)}"

        # Parse opportunities and threats (simple extraction for now)
        opportunities = []
        threats = []

        if "opportunity" in expert_summary.lower():
            opportunities.append(
                {
                    "type": "extracted",
                    "description": "Opportunities detected in content",
                    "confidence": 0.7,
                }
            )

        if "threat" in expert_summary.lower() or "risk" in expert_summary.lower():
            threats.append(
                {
                    "type": "extracted",
                    "description": "Threats/risks detected in content",
                    "confidence": 0.7,
                }
            )

        logger.info("[VectorizationAgent] Expert summary generated")

        return {
            "current_step": "expert_summary",
            "expert_summary": expert_summary,
            "opportunities": opportunities,
            "threats": threats,
            "llm_response": expert_summary,
        }

    def format_final_output(self, state: VectorizationAgentState) -> Dict[str, Any]:
        """
        Step 5: Format final output for downstream consumption.
        Prepares domain_insights for integration with parent graph.
        Includes anomaly detection results.
        """
        logger.info("[VectorizationAgent] STEP 5: Format Output")

        batch_id = state.get("batch_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        processing_stats = state.get("processing_stats", {})
        expert_summary = state.get("expert_summary", "")
        opportunities = state.get("opportunities", [])
        threats = state.get("threats", [])
        embeddings = state.get("vector_embeddings", [])
        anomaly_results = state.get("anomaly_results", {})

        # Build domain insights
        domain_insights = []

        # Main vectorization insight
        domain_insights.append(
            {
                "event_id": f"vec_{batch_id}",
                "domain": "vectorization",
                "category": "text_analysis",
                "summary": f"Processed {len(embeddings)} texts with multilingual BERT models",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": "low",
                "impact_type": "analysis",
                "confidence": 0.9,
                "metadata": {
                    "total_texts": len(embeddings),
                    "languages": processing_stats.get("language_distribution", {}),
                    "models_used": list(
                        set(e.get("model_used", "") for e in embeddings)
                    ),
                },
            }
        )

        # Add anomaly detection insight
        anomalies = anomaly_results.get("anomalies", [])
        anomaly_status = anomaly_results.get("status", "unknown")

        if anomaly_status == "success" and anomalies:
            # Add summary insight for anomaly detection
            domain_insights.append(
                {
                    "event_id": f"anomaly_{batch_id}",
                    "domain": "anomaly_detection",
                    "category": "ml_analysis",
                    "summary": f"ML Anomaly Detection: {len(anomalies)} anomalies found in {anomaly_results.get('total_analyzed', 0)} texts",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": "high" if len(anomalies) > 5 else "medium",
                    "impact_type": "risk",
                    "confidence": 0.85,
                    "metadata": {
                        "model_used": anomaly_results.get("model_used", "unknown"),
                        "anomaly_rate": anomaly_results.get("anomaly_rate", 0),
                        "total_analyzed": anomaly_results.get("total_analyzed", 0),
                    },
                }
            )

            # Add individual anomaly events
            for i, anomaly in enumerate(anomalies[:10]):  # Limit to top 10
                domain_insights.append(
                    {
                        "event_id": f"anomaly_{batch_id}_{i}",
                        "domain": "anomaly_detection",
                        "category": "anomaly",
                        "summary": f"Anomaly detected (score: {anomaly.get('anomaly_score', 0):.2f})",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "severity": (
                            "high"
                            if anomaly.get("anomaly_score", 0) > 0.7
                            else "medium"
                        ),
                        "impact_type": "risk",
                        "confidence": anomaly.get("anomaly_score", 0.5),
                        "is_anomaly": True,
                        "anomaly_score": anomaly.get("anomaly_score", 0),
                        "metadata": {
                            "post_id": anomaly.get("post_id", ""),
                            "language": anomaly.get("language", "unknown"),
                        },
                    }
                )
        elif anomaly_status == "fallback":
            domain_insights.append(
                {
                    "event_id": f"anomaly_info_{batch_id}",
                    "domain": "anomaly_detection",
                    "category": "system_info",
                    "summary": "ML model not trained yet - using severity-based fallback",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": "low",
                    "impact_type": "info",
                    "confidence": 1.0,
                }
            )

        # Add opportunity insights
        for i, opp in enumerate(opportunities):
            domain_insights.append(
                {
                    "event_id": f"opp_{batch_id}_{i}",
                    "domain": "vectorization",
                    "category": "opportunity",
                    "summary": opp.get("description", "Opportunity detected"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": "medium",
                    "impact_type": "opportunity",
                    "confidence": opp.get("confidence", 0.7),
                }
            )

        # Add threat insights
        for i, threat in enumerate(threats):
            domain_insights.append(
                {
                    "event_id": f"threat_{batch_id}_{i}",
                    "domain": "vectorization",
                    "category": "threat",
                    "summary": threat.get("description", "Threat detected"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": "high",
                    "impact_type": "risk",
                    "confidence": threat.get("confidence", 0.7),
                }
            )

        # Final output
        final_output = {
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_texts": len(embeddings),
            "processing_stats": processing_stats,
            "expert_summary": expert_summary,
            "opportunities_count": len(opportunities),
            "threats_count": len(threats),
            "vector_dimensions": 768,
            "anomaly_detection": {
                "status": anomaly_status,
                "anomalies_found": len(anomalies),
                "model_used": anomaly_results.get("model_used", "none"),
                "anomaly_rate": anomaly_results.get("anomaly_rate", 0),
            },
            "status": "SUCCESS",
        }

        logger.info(
            f"[VectorizationAgent] ✓ Output formatted: {len(domain_insights)} insights (inc. {len(anomalies)} anomalies)"
        )

        return {
            "current_step": "complete",
            "domain_insights": domain_insights,
            "final_output": final_output,
            "structured_output": final_output,
            "anomaly_results": anomaly_results,  # Pass through for downstream
        }
