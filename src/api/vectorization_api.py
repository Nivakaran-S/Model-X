"""
src/api/vectorization_api.py
FastAPI endpoint for the Vectorization Agent
Production-grade API for text-to-vector conversion
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import uvicorn

from src.graphs.vectorizationAgentGraph import graph as vectorization_graph

logger = logging.getLogger("vectorization_api")

# Create FastAPI app
app = FastAPI(
    title="Roger Vectorization Agent API",
    description="API for converting multilingual text to vectors using language-specific BERT models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TextInput(BaseModel):
    """Single text input for vectorization"""
    text: str = Field(..., description="Text content to vectorize")
    post_id: Optional[str] = Field(None, description="Unique identifier for the text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class VectorizationRequest(BaseModel):
    """Request for batch text vectorization"""
    texts: List[TextInput] = Field(..., description="List of texts to vectorize")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    include_vectors: bool = Field(True, description="Include full vectors in response")
    include_expert_summary: bool = Field(True, description="Generate LLM expert summary")


class VectorizationResponse(BaseModel):
    """Response from vectorization"""
    batch_id: str
    status: str
    total_processed: int
    language_distribution: Dict[str, int]
    expert_summary: Optional[str]
    opportunities_count: int
    threats_count: int
    domain_insights: List[Dict[str, Any]]
    processing_time_seconds: float
    vectors: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    vectorizer_available: bool
    llm_available: bool


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from src.llms.groqllm import GroqLLM
    
    try:
        llm = GroqLLM().get_llm()
        llm_available = True
    except Exception:
        llm_available = False
    
    try:
        from models.anomaly_detection.src.utils import get_vectorizer
        vectorizer = get_vectorizer()
        vectorizer_available = True
    except Exception:
        vectorizer_available = False
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        vectorizer_available=vectorizer_available,
        llm_available=llm_available
    )


@app.post("/vectorize", response_model=VectorizationResponse)
async def vectorize_texts(request: VectorizationRequest):
    """
    Vectorize a batch of texts using language-specific BERT models.
    
    Steps:
    1. Language Detection (FastText/lingua-py)
    2. Text Vectorization (SinhalaBERTo/Tamil-BERT/DistilBERT)
    3. Expert Summary (GroqLLM - optional)
    4. Opportunity/Threat Analysis
    """
    start_time = datetime.utcnow()
    
    try:
        # Prepare input
        input_texts = []
        for i, text_input in enumerate(request.texts):
            input_texts.append({
                "text": text_input.text,
                "post_id": text_input.post_id or f"text_{i}",
                "metadata": text_input.metadata or {}
            })
        
        batch_id = request.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run vectorization graph
        initial_state = {
            "input_texts": input_texts,
            "batch_id": batch_id
        }
        
        result = vectorization_graph.invoke(initial_state)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Build response
        final_output = result.get("final_output", {})
        processing_stats = result.get("processing_stats", {})
        
        response = VectorizationResponse(
            batch_id=batch_id,
            status="SUCCESS",
            total_processed=final_output.get("total_texts", len(input_texts)),
            language_distribution=processing_stats.get("language_distribution", {}),
            expert_summary=result.get("expert_summary") if request.include_expert_summary else None,
            opportunities_count=final_output.get("opportunities_count", 0),
            threats_count=final_output.get("threats_count", 0),
            domain_insights=result.get("domain_insights", []),
            processing_time_seconds=processing_time,
            vectors=result.get("vector_embeddings") if request.include_vectors else None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-language")
async def detect_language(texts: List[str]):
    """
    Detect language for a list of texts.
    Returns language code (en/si/ta) and confidence for each text.
    """
    try:
        from models.anomaly_detection.src.utils import detect_language as detect_lang
        
        results = []
        for text in texts:
            lang, conf = detect_lang(text)
            results.append({
                "text_preview": text[:100],
                "language": lang,
                "confidence": conf
            })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available language-specific models"""
    return {
        "models": {
            "english": {
                "name": "DistilBERT",
                "hf_name": "distilbert-base-uncased",
                "description": "Fast and accurate English understanding"
            },
            "sinhala": {
                "name": "SinhalaBERTo",
                "hf_name": "keshan/SinhalaBERTo",
                "description": "Specialized Sinhala context and sentiment"
            },
            "tamil": {
                "name": "Tamil-BERT",
                "hf_name": "l3cube-pune/tamil-bert",
                "description": "Specialized Tamil understanding"
            }
        },
        "language_detection": {
            "primary": "FastText (lid.176.bin)",
            "fallback": "lingua-py + Unicode script detection"
        },
        "vector_dimension": 768
    }


# ============================================================================
# RUN SERVER
# ============================================================================

def start_vectorization_server(host: str = "0.0.0.0", port: int = 8001):
    """Start the FastAPI server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_vectorization_server()
