"""
main.py
Production-Ready Real-Time Intelligence Platform Backend
- Uses combinedAgentGraph for multi-agent orchestration
- Threading for concurrent graph execution and WebSocket server
- Database-driven feed updates with polling
- Duplicate prevention
- District-based feed categorization for map display

Updated: Resilient WebSocket handling for long scraping operations (60s+ cycles)
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Set, Optional
import asyncio
import json
from datetime import datetime
import sys
import os
import logging
import threading
import time
import uuid  # CRITICAL: Was missing, needed for event_id generation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.graphs.combinedAgentGraph import graph
from src.states.combinedAgentState import CombinedAgentState
from src.storage.storage_manager import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Roger_api")


# ============================================
# AUTO-TRAINING: Check and train models if missing
# ============================================

def check_and_train_models():
    """
    Check if ML models are trained. If not, trigger training in background.
    Called on startup to ensure models are available.
    """
    from pathlib import Path
    import subprocess
    
    PROJECT_ROOT = Path(__file__).parent
    
    # Define model checks: (name, model_path, train_command)
    model_checks = [
        {
            "name": "Anomaly Detection",
            "check_paths": [
                PROJECT_ROOT / "models" / "anomaly-detection" / "artifacts" / "models",
            ],
            "check_files": ["*.joblib", "*.pkl"],
            "train_cmd": [sys.executable, str(PROJECT_ROOT / "models" / "anomaly-detection" / "main.py")]
        },
        {
            "name": "Weather Prediction",
            "check_paths": [
                PROJECT_ROOT / "models" / "weather-prediction" / "artifacts" / "models",
            ],
            "check_files": ["*.h5", "*.keras"],
            "train_cmd": [sys.executable, str(PROJECT_ROOT / "models" / "weather-prediction" / "main.py"), "--mode", "full"]
        },
        {
            "name": "Currency Prediction",
            "check_paths": [
                PROJECT_ROOT / "models" / "currency-volatility-prediction" / "artifacts" / "models",
            ],
            "check_files": ["*.h5", "*.keras"],
            "train_cmd": [sys.executable, str(PROJECT_ROOT / "models" / "currency-volatility-prediction" / "main.py"), "--mode", "full"]
        },
        {
            "name": "Stock Prediction",
            "check_paths": [
                PROJECT_ROOT / "models" / "stock-price-prediction" / "artifacts" / "models",
            ],
            "check_files": ["*.h5", "*.keras"],
            "train_cmd": [sys.executable, str(PROJECT_ROOT / "models" / "stock-price-prediction" / "main.py"), "--mode", "full"]
        },
    ]
    
    def has_trained_model(check_paths, check_files):
        """Check if any trained model files exist."""
        for path in check_paths:
            if path.exists():
                for pattern in check_files:
                    if list(path.glob(pattern)):
                        return True
                    # Also check subdirectories
                    if list(path.glob(f"**/{pattern}")):
                        return True
        return False
    
    def train_in_background(name, cmd):
        """Run training in a background thread."""
        def _train():
            logger.info(f"[AUTO-TRAIN] Starting {name} training...")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 min timeout
                )
                if result.returncode == 0:
                    logger.info(f"[AUTO-TRAIN] ✓ {name} training complete!")
                else:
                    logger.warning(f"[AUTO-TRAIN] ⚠ {name} training failed: {result.stderr[:500]}")
            except subprocess.TimeoutExpired:
                logger.error(f"[AUTO-TRAIN] ✗ {name} training timed out (30 min)")
            except Exception as e:
                logger.error(f"[AUTO-TRAIN] ✗ {name} training error: {e}")
        
        thread = threading.Thread(target=_train, daemon=True, name=f"train_{name}")
        thread.start()
        return thread
    
    # Check each model
    training_threads = []
    for model in model_checks:
        if has_trained_model(model["check_paths"], model["check_files"]):
            logger.info(f"[MODEL CHECK] ✓ {model['name']} - Model found")
        else:
            logger.warning(f"[MODEL CHECK] ⚠ {model['name']} - No model found, starting training...")
            thread = train_in_background(model["name"], model["train_cmd"])
            training_threads.append((model["name"], thread))
    
    if training_threads:
        logger.info(f"[AUTO-TRAIN] Started {len(training_threads)} background training jobs")
    else:
        logger.info("[MODEL CHECK] All models found - no training needed")
    
    return training_threads


# Run model check on module load (startup)
logger.info("=" * 60)
logger.info("[STARTUP] Checking ML models...")
logger.info("=" * 60)
_training_threads = check_and_train_models()

app = FastAPI(title="Roger Intelligence Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_state: Dict[str, Any] = {
    "final_ranked_feed": [],
    "risk_dashboard_snapshot": {
        "logistics_friction": 0.0,
        "compliance_volatility": 0.0,
        "market_instability": 0.0,
        "opportunity_index": 0.0,
        "avg_confidence": 0.0,
        "high_priority_count": 0,
        "total_events": 0,
        "last_updated": datetime.utcnow().isoformat()
    },
    "run_count": 0,
    "status": "initializing",
    "first_run_complete": False  # Track first graph execution
}

# Thread-safe communication
feed_update_queue = asyncio.Queue()
seen_event_ids: Set[str] = set()  # Duplicate prevention

# Global event loop reference for cross-thread broadcasting
main_event_loop = None

# Storage manager
storage_manager = StorageManager()

# WebSocket settings - RESILIENT for long scraping operations (60s+ graph cycles)
# Increased intervals to prevent disconnections during lengthy scraping
HEARTBEAT_INTERVAL = 45.0  # Send ping every 45s (was 25s)
HEARTBEAT_TIMEOUT = 30.0   # Wait 30s for pong (was 10s) 
HEARTBEAT_MISS_THRESHOLD = 4  # Allow 4 misses (was 3) = ~3 minutes tolerance
SEND_TIMEOUT = 10.0  # Increased from 5s

class ConnectionManager:
    """Manages active WebSocket with heartbeat"""
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            meta = {
                "heartbeat_task": asyncio.create_task(self._heartbeat_loop(websocket)),
                "last_pong": datetime.utcnow(),
                "misses": 0
            }
            self.active_connections[websocket] = meta
            logger.info(f"[WebSocket] Connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            meta = self.active_connections.pop(websocket, None)
        if meta:
            task = meta.get("heartbeat_task")
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            try:
                await websocket.close()
            except Exception:
                pass
            logger.info(f"[WebSocket] Disconnected. Total: {len(self.active_connections)}")

    async def _send_with_timeout(self, websocket: WebSocket, message_json: str):
        try:
            await asyncio.wait_for(websocket.send_text(message_json), timeout=SEND_TIMEOUT)
            return True
        except Exception as e:
            logger.debug(f"[WebSocket] Send failed: {e}")
            return False

    async def _heartbeat_loop(self, websocket: WebSocket):
        """Per-connection heartbeat task"""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if websocket not in self.active_connections:
                    break

                ping_payload = json.dumps({"type": "ping"})
                ok = await self._send_with_timeout(websocket, ping_payload)
                if not ok:
                    async with self._lock:
                        meta = self.active_connections.get(websocket)
                        if meta is not None:
                            meta['misses'] += 1
                else:
                    waited = 0.0
                    sleep_step = 0.5
                    pong_received = False
                    while waited < HEARTBEAT_TIMEOUT:
                        await asyncio.sleep(sleep_step)
                        waited += sleep_step
                        async with self._lock:
                            meta = self.active_connections.get(websocket)
                            if meta is None:
                                return
                            last_pong = meta.get("last_pong")
                            if last_pong and (datetime.utcnow() - last_pong).total_seconds() < (HEARTBEAT_INTERVAL + HEARTBEAT_TIMEOUT):
                                pong_received = True
                                meta['misses'] = 0
                                break
                    if not pong_received:
                        async with self._lock:
                            meta = self.active_connections.get(websocket)
                            if meta is not None:
                                meta['misses'] += 1

                async with self._lock:
                    meta = self.active_connections.get(websocket)
                    if meta is None:
                        return
                    if meta.get('misses', 0) >= HEARTBEAT_MISS_THRESHOLD:
                        logger.warning("[WebSocket] Miss threshold exceeded, disconnecting")
                        try:
                            await websocket.close(code=1001)
                        except Exception:
                            pass
                        await self.disconnect(websocket)
                        return

        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"[WebSocket] Heartbeat error: {e}")
            try:
                await self.disconnect(websocket)
            except Exception:
                pass

    async def broadcast(self, message: dict):
        """Broadcast to all connections"""
        async with self._lock:
            conns = list(self.active_connections.keys())
        if not conns:
            return
        message_json = json.dumps(message, default=str)
        dead: List[WebSocket] = []
        for conn in conns:
            ok = await self._send_with_timeout(conn, message_json)
            if not ok:
                dead.append(conn)
        for conn in dead:
            logger.info("[WebSocket] Removing dead connection")
            await self.disconnect(conn)

manager = ConnectionManager()


def categorize_feed_by_district(feed: Dict[str, Any]) -> str:
    """
    Categorize feed by Sri Lankan district based on summary text.
    Returns district name or "National" if not district-specific.
    NOTE: This returns the FIRST match. Use get_all_matching_districts() for multi-district feeds.
    """
    districts = get_all_matching_districts(feed)
    return districts[0] if districts else "National"


def get_all_matching_districts(feed: Dict[str, Any]) -> List[str]:
    """
    Get ALL districts mentioned in a feed (direct or via province).
    
    Supports:
    - Direct district names (Colombo, Kandy, etc.)
    - Province names that map to multiple districts
    - Commonly referenced regions
    
    Returns list of all matching district names.
    """
    summary = feed.get("summary", "").lower()
    
    # Sri Lankan districts
    districts = [
        "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
        "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
        "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
        "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
        "Moneragala", "Ratnapura", "Kegalle"
    ]
    
    # Province to districts mapping
    province_mapping = {
        "western province": ["Colombo", "Gampaha", "Kalutara"],
        "western": ["Colombo", "Gampaha", "Kalutara"],
        "central province": ["Kandy", "Matale", "Nuwara Eliya"],
        "central": ["Kandy", "Matale", "Nuwara Eliya"],
        "southern province": ["Galle", "Matara", "Hambantota"],
        "southern provinces": ["Galle", "Matara", "Hambantota"],
        "southern": ["Galle", "Matara", "Hambantota"],
        "south": ["Galle", "Matara", "Hambantota"],
        "northern province": ["Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu"],
        "northern": ["Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu"],
        "north": ["Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu"],
        "eastern province": ["Batticaloa", "Ampara", "Trincomalee"],
        "eastern": ["Batticaloa", "Ampara", "Trincomalee"],
        "east": ["Batticaloa", "Ampara", "Trincomalee"],
        "north western province": ["Kurunegala", "Puttalam"],
        "north western": ["Kurunegala", "Puttalam"],
        "north central province": ["Anuradhapura", "Polonnaruwa"],
        "north central": ["Anuradhapura", "Polonnaruwa"],
        "uva province": ["Badulla", "Moneragala"],
        "uva": ["Badulla", "Moneragala"],
        "sabaragamuwa province": ["Ratnapura", "Kegalle"],
        "sabaragamuwa": ["Ratnapura", "Kegalle"],
    }
    
    matched_districts = set()
    
    # Check for province mentions first
    for province, province_districts in province_mapping.items():
        if province in summary:
            matched_districts.update(province_districts)
    
    # Check for direct district mentions
    for district in districts:
        if district.lower() in summary:
            matched_districts.add(district)
    
    return list(matched_districts)


def run_graph_loop():
    """
    Graph execution in separate thread.
    Runs the combinedAgentGraph and stores results in database.
    """
    logger.info("="*80)
    logger.info("[GRAPH THREAD] Starting Roger combinedAgentGraph loop")
    logger.info("="*80)
    
    initial_state = CombinedAgentState(
        domain_insights=[],
        final_ranked_feed=[],
        run_count=0,
        max_runs=999,  # Continuous mode
        route=None
    )
    
    try:
        # Note: Using synchronous invoke since we're in a thread
        # Increase recursion limit for the multi-agent graph (default is 25)
        config = {"recursion_limit": 100}
        for event in graph.stream(initial_state, config=config):
            logger.info(f"[GRAPH] Event nodes: {list(event.keys())}")
            
            for node_name, node_output in event.items():
                # Extract feed data
                if hasattr(node_output, 'final_ranked_feed'):
                    feeds = node_output.final_ranked_feed
                elif isinstance(node_output, dict):
                    feeds = node_output.get('final_ranked_feed', [])
                else:
                    continue
                
                if feeds:
                    logger.info(f"[GRAPH] {node_name} produced {len(feeds)} feeds")
                    
                    # FIELD_NORMALIZATION: Transform graph format to frontend format
                    for feed_item in feeds:
                        if isinstance(feed_item, dict):
                            event_data = feed_item
                        else:
                            event_data = feed_item.__dict__ if hasattr(feed_item, '__dict__') else {}
                        
                        # Normalize field names: graph uses content_summary/target_agent, frontend expects summary/domain
                        event_id = event_data.get("event_id", str(uuid.uuid4()))
                        summary = event_data.get("content_summary") or event_data.get("summary", "")
                        domain = event_data.get("target_agent") or event_data.get("domain", "unknown")
                        severity = event_data.get("severity", "medium")
                        impact_type = event_data.get("impact_type", "risk")
                        confidence = event_data.get("confidence_score", event_data.get("confidence", 0.5))
                        timestamp = event_data.get("timestamp", datetime.utcnow().isoformat())
                        
                        # Check for duplicates
                        is_dup, _, _ = storage_manager.is_duplicate(summary)
                        
                        if not is_dup:
                            try:
                                storage_manager.store_event(
                                    event_id=event_id,
                                    summary=summary,
                                    domain=domain,
                                    severity=severity,
                                    impact_type=impact_type,
                                    confidence_score=confidence
                                )
                                logger.info(f"[GRAPH] Stored new feed: {summary[:60]}...")
                            except Exception as storage_error:
                                logger.warning(f"[GRAPH] Storage error (continuing): {storage_error}")
                            
                            # DIRECT_BROADCAST_FIX: Set first_run_complete and broadcast
                            if not current_state.get('first_run_complete'):
                                current_state['first_run_complete'] = True
                                current_state['status'] = 'operational'
                                logger.info("[GRAPH] FIRST RUN COMPLETE - Broadcasting to frontend!")
                                
                                # Trigger broadcast from sync thread to async loop
                                if main_event_loop:
                                    asyncio.run_coroutine_threadsafe(
                                        manager.broadcast(current_state),
                                        main_event_loop
                                    )
                
                # Small delay to prevent CPU overload
                time.sleep(0.3)
                
    except Exception as e:
        logger.error(f"[GRAPH THREAD] Error: {e}", exc_info=True)


async def database_polling_loop():
    """
    Polls database for new feeds and broadcasts via WebSocket.
    Runs concurrently with graph thread.
    """
    global current_state
    last_check = datetime.utcnow()
    
    logger.info("[DB_POLLER] Starting database polling loop")
    
    while True:
        try:
            await asyncio.sleep(2.0)  # Poll every 2 seconds
            
            # Get new feeds since last check
            new_feeds = storage_manager.get_feeds_since(last_check)
            last_check = datetime.utcnow()
            
            if new_feeds:
                logger.info(f"[DB_POLLER] Found {len(new_feeds)} new feeds")
                
                # Filter duplicates (by event_id)
                unique_feeds = []
                for feed in new_feeds:
                    event_id = feed.get("event_id")
                    if event_id and event_id not in seen_event_ids:
                        seen_event_ids.add(event_id)
                        
                        # Add district categorization for map
                        feed["district"] = categorize_feed_by_district(feed)
                        unique_feeds.append(feed)
                
                if unique_feeds:
                    # Update current state
                    current_state['final_ranked_feed'] = unique_feeds + current_state.get('final_ranked_feed', [])
                    current_state['final_ranked_feed'] = current_state['final_ranked_feed'][:100]  # Keep last 100
                    current_state['status'] = 'operational'
                    current_state['last_update'] = datetime.utcnow().isoformat()
                    
                    # Mark first run as complete (frontend loading screen can now hide)
                    if not current_state.get('first_run_complete'):
                        current_state['first_run_complete'] = True
                        logger.info("[DB_POLLER] First graph run complete! Frontend loading screen can now hide.")
                    
                    # Broadcast to WebSocket clients
                    await manager.broadcast(current_state)
                    logger.info(f"[DB_POLLER] Broadcasted {len(unique_feeds)} unique feeds")
            
        except Exception as e:
            logger.error(f"[DB_POLLER] Error: {e}")



@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    
    logger.info("[API] Starting Roger API...")
    
    # Start graph execution in separate thread
    graph_thread = threading.Thread(target=run_graph_loop, daemon=True)
    graph_thread.start()
    logger.info("[API] Graph thread started")
    
    # Start database polling loop
    asyncio.create_task(database_polling_loop())
    logger.info("[API] Database polling started")


@app.get("/")
def read_root():
    return {
        "service": "Roger Intelligence Platform",
        "status": current_state.get("status"),
        "version": "2.0.0 (Database-Driven)"
    }


@app.get("/api/status")
def get_status():
    return {
        "status": current_state.get("status"),
        "run_count": current_state.get("run_count"),
        "last_update": current_state.get("last_update"),
        "active_connections": len(manager.active_connections),
        "total_events": len(current_state.get("final_ranked_feed", []))
    }


@app.get("/api/dashboard")
def get_dashboard():
    return current_state.get("risk_dashboard_snapshot", {})


@app.get("/api/feed")
def get_feed():
    """Get current feed from memory"""
    return {
        "events": current_state.get("final_ranked_feed", []),
        "total": len(current_state.get("final_ranked_feed", []))
    }


@app.get("/api/feeds")
def get_feeds_from_db(limit: int = 100):
    """Get feeds directly from database (for initial load)"""
    try:
        feeds = storage_manager.get_recent_feeds(limit=limit)
        
        # FIELD_NORMALIZATION + district categorization
        normalized_feeds = []
        for feed in feeds:
            # Ensure frontend-compatible field names
            normalized = {
                "event_id": feed.get("event_id"),
                "summary": feed.get("summary", ""),
                "domain": feed.get("domain", "unknown"),
                "severity": feed.get("severity", "medium"),
                "impact_type": feed.get("impact_type", "risk"),
                "confidence": feed.get("confidence", 0.5),
                "timestamp": feed.get("timestamp"),
                "district": categorize_feed_by_district(feed)
            }
            normalized_feeds.append(normalized)
        
        return {
            "events": normalized_feeds,
            "total": len(normalized_feeds),
            "source": "database"
        }
    except Exception as e:
        logger.error(f"[API] Error fetching feeds: {e}")
        return {"events": [], "total": 0, "error": str(e)}


@app.get("/api/feeds/by_district/{district}")
def get_feeds_by_district(district: str, limit: int = 50):
    """Get feeds for specific district"""
    try:
        all_feeds = storage_manager.get_recent_feeds(limit=200)
        
        # Filter by district
        district_feeds = []
        for feed in all_feeds:
            feed["district"] = categorize_feed_by_district(feed)
            if feed["district"].lower() == district.lower():
                district_feeds.append(feed)
                if len(district_feeds) >= limit:
                    break
        
        return {
            "district": district,
            "events": district_feeds,
            "total": len(district_feeds)
        }
    except Exception as e:
        logger.error(f"[API] Error fetching district feeds: {e}")
        return {"events": [], "total": 0, "error": str(e)}


@app.get("/api/rivernet")
def get_rivernet_status():
    """Get real-time river monitoring data from RiverNet.lk"""
    try:
        from src.utils.utils import tool_rivernet_status
        river_data = tool_rivernet_status()
        return river_data
    except Exception as e:
        logger.error(f"[API] Error fetching rivernet data: {e}")
        return {
            "rivers": [],
            "alerts": [],
            "summary": {"total_monitored": 0, "overall_status": "error", "has_alerts": False},
            "error": str(e)
        }


@app.get("/api/weather/historical")
def get_historical_climate_data():
    """
    Get 30-year historical flood pattern analysis.
    
    Returns climate trend data including:
    - Average annual rainfall
    - Maximum daily rainfall records
    - Heavy/extreme rain day counts
    - Decadal comparison (1995-2025)
    - Key climate change findings
    """
    try:
        from src.utils.utils import tool_floodwatch_historical
        historical_data = tool_floodwatch_historical()
        return {
            "status": "success",
            "data": historical_data
        }
    except Exception as e:
        logger.error(f"[API] Error fetching historical data: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/api/weather/threat")
def get_national_threat_score():
    """
    Get national flood threat score (0-100).
    
    Aggregates river status, DMC alerts, and seasonal factors
    to compute an overall threat level for Sri Lanka.
    
    Returns:
    - national_threat_score (0-100)
    - threat_level (CRITICAL/HIGH/MODERATE/LOW)
    - breakdown by category
    - risk district lists
    """
    try:
        from src.utils.utils import tool_rivernet_status, tool_calculate_national_threat, tool_dmc_alerts
        
        # Get river data
        river_data = None
        try:
            river_data = tool_rivernet_status()
        except Exception as e:
            logger.warning(f"[ThreatAPI] RiverNet unavailable: {e}")
        
        # Get DMC alerts
        dmc_data = None
        try:
            dmc_result = tool_dmc_alerts()
            dmc_data = dmc_result.get("alerts", [])
        except Exception as e:
            logger.warning(f"[ThreatAPI] DMC unavailable: {e}")
        
        # Calculate threat score
        threat_data = tool_calculate_national_threat(
            river_data=river_data,
            dmc_alerts=dmc_data
        )
        
        return {
            "status": "success",
            **threat_data
        }
    except Exception as e:
        logger.error(f"[API] Error calculating threat: {e}")
        return {
            "status": "error",
            "national_threat_score": 0,
            "threat_level": "UNKNOWN",
            "error": str(e)
        }


@app.get("/api/weather/predictions")
def get_weather_predictions():
    """
    Get next-day weather predictions for all 25 Sri Lankan districts.
    
    Returns predictions from trained LSTM models (or climate fallback if models not available).
    Includes temperature, rainfall, humidity, flood risk, and severity for each district.
    """
    try:
        from pathlib import Path
        import json
        from datetime import datetime, timedelta
        
        # Path to predictions output
        predictions_dir = Path(__file__).parent / "models" / "weather-prediction" / "output" / "predictions"
        
        # Try to find most recent predictions file
        prediction_files = list(predictions_dir.glob("predictions_*.json")) if predictions_dir.exists() else []
        
        if prediction_files:
            # Get most recent predictions file
            latest_file = max(prediction_files, key=lambda p: p.stem)
            
            with open(latest_file, "r") as f:
                predictions = json.load(f)
            
            return {
                "status": "success",
                "prediction_date": predictions.get("prediction_date", ""),
                "generated_at": predictions.get("generated_at", ""),
                "districts": predictions.get("districts", {}),
                "total_districts": len(predictions.get("districts", {})),
                "source": "lstm_models" if not predictions.get("is_fallback") else "climate_fallback"
            }
        
        # No predictions file - try to generate on-the-fly
        try:
            from models.weather_prediction.src.components.predictor import WeatherPredictor
            
            predictor = WeatherPredictor()
            predictions = predictor.predict_all_districts()
            
            return {
                "status": "success",
                "prediction_date": predictions.get("prediction_date", (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")),
                "generated_at": predictions.get("generated_at", datetime.now().isoformat()),
                "districts": predictions.get("districts", {}),
                "total_districts": len(predictions.get("districts", {})),
                "source": "live_prediction"
            }
        except Exception as pred_err:
            logger.warning(f"[WeatherAPI] Could not generate live predictions: {pred_err}")
        
        # Fallback - no predictions available
        return {
            "status": "no_data",
            "message": "Weather predictions not available. Run: python models/weather-prediction/main.py --mode predict",
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "districts": {},
            "total_districts": 0
        }
        
    except Exception as e:
        logger.error(f"[WeatherAPI] Error fetching predictions: {e}")
        return {
            "status": "error",
            "error": str(e),
            "districts": {},
            "total_districts": 0
        }


# ============================================
# CURRENCY PREDICTION ENDPOINTS
# ============================================

@app.get("/api/currency/prediction")
def get_currency_prediction():
    """
    Get next-day USD/LKR currency prediction.
    
    Returns prediction from trained GRU model (or fallback if model not available).
    """
    try:
        from pathlib import Path
        import json
        from datetime import datetime, timedelta
        
        # Path to currency predictions output
        predictions_dir = Path(__file__).parent / "models" / "currency-volatility-prediction" / "output" / "predictions"
        
        # Try to find most recent predictions file
        prediction_files = list(predictions_dir.glob("currency_prediction_*.json")) if predictions_dir.exists() else []
        
        if prediction_files:
            # Get most recent predictions file
            latest_file = max(prediction_files, key=lambda p: p.stem)
            
            with open(latest_file, "r") as f:
                prediction = json.load(f)
            
            return {
                "status": "success",
                "prediction": prediction,
                "source": "gru_model" if not prediction.get("is_fallback") else "fallback"
            }
        
        # No predictions file
        return {
            "status": "no_data",
            "message": "Currency prediction not available. Run: python models/currency-volatility-prediction/main.py --mode predict",
            "prediction": None
        }
        
    except Exception as e:
        logger.error(f"[CurrencyAPI] Error fetching prediction: {e}")
        return {
            "status": "error",
            "error": str(e),
            "prediction": None
        }


@app.get("/api/currency/history")
def get_currency_history(days: int = 7):
    """
    Get historical USD/LKR exchange rate data.
    
    Args:
        days: Number of days of history to return (default 7)
    
    Returns:
        List of historical rates with date and close price.
    """
    try:
        from pathlib import Path
        import pandas as pd
        
        # Path to currency data
        data_dir = Path(__file__).parent / "models" / "currency-volatility-prediction" / "artifacts" / "data"
        
        # Find the data file
        data_files = list(data_dir.glob("currency_data_*.csv")) if data_dir.exists() else []
        
        if data_files:
            # Get most recent data file
            latest_file = max(data_files, key=lambda p: p.stem)
            df = pd.read_csv(latest_file)
            
            # Get last N days
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).head(days)
            df = df.sort_values('date', ascending=True)
            
            history = []
            for _, row in df.iterrows():
                history.append({
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "close": float(row['close']),
                    "high": float(row.get('high', row['close'])),
                    "low": float(row.get('low', row['close']))
                })
            
            return {
                "status": "success",
                "history": history,
                "days": len(history)
            }
        
        return {
            "status": "no_data",
            "message": "No historical data available. Run data ingestion first.",
            "history": []
        }
        
    except Exception as e:
        logger.error(f"[CurrencyAPI] Error fetching history: {e}")
        return {
            "status": "error",
            "error": str(e),
            "history": []
        }


# ============================================
# TRENDING DETECTION ENDPOINTS
# ============================================

@app.get("/api/trending")
def get_trending_topics(limit: int = 10):
    """
    Get currently trending topics.
    
    Returns topics with momentum > 2x (gaining traction).
    """
    try:
        from src.utils.trending_detector import get_trending_now, get_spikes
        
        trending = get_trending_now(limit=limit)
        spikes = get_spikes()
        
        return {
            "status": "success",
            "trending_topics": trending,
            "spike_alerts": spikes,
            "total_trending": len(trending),
            "total_spikes": len(spikes)
        }
        
    except Exception as e:
        logger.error(f"[TrendingAPI] Error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "trending_topics": [],
            "spike_alerts": []
        }


@app.get("/api/trending/topic/{topic}")
def get_topic_history(topic: str, hours: int = 24):
    """
    Get hourly mention history for a specific topic.
    
    Args:
        topic: Topic name to get history for
        hours: Number of hours of history to return (default 24)
    """
    try:
        from src.utils.trending_detector import get_trending_detector
        
        detector = get_trending_detector()
        history = detector.get_topic_history(topic, hours=hours)
        momentum = detector.get_momentum(topic)
        is_spike = detector.is_spike(topic)
        
        return {
            "status": "success",
            "topic": topic,
            "momentum": momentum,
            "is_spike": is_spike,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"[TrendingAPI] Error getting history for {topic}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "topic": topic,
            "momentum": 1.0,
            "is_spike": False,
            "history": []
        }


@app.post("/api/trending/record")
def record_topic_mention(topic: str, source: str = "manual", domain: str = "general"):
    """
    Record a topic mention (for testing/manual tracking).
    
    Args:
        topic: Topic/keyword being mentioned
        source: Source of the mention (twitter, news, etc.)
        domain: Domain category (political, economical, etc.)
    """
    try:
        from src.utils.trending_detector import record_topic_mention as record_mention
        
        record_mention(topic=topic, source=source, domain=domain)
        
        # Get updated momentum
        from src.utils.trending_detector import get_trending_detector
        detector = get_trending_detector()
        momentum = detector.get_momentum(topic)
        
        return {
            "status": "success",
            "message": f"Recorded mention for '{topic}'",
            "current_momentum": momentum,
            "is_spike": detector.is_spike(topic)
        }
        
    except Exception as e:
        logger.error(f"[TrendingAPI] Error recording mention: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================
# ANOMALY DETECTION ENDPOINTS
# ============================================

# Lazy-loaded anomaly detection components
_anomaly_model = None
_vectorizer = None
_language_detector = None


def _load_anomaly_components():
    """Load anomaly detection model and vectorizer"""
    global _anomaly_model, _vectorizer, _language_detector
    
    if _anomaly_model is not None:
        return True
    
    try:
        import joblib
        from pathlib import Path
        
        # Model path
        models_dir = Path(__file__).parent / "models" / "anomaly-detection" / "src" / "components"
        output_dir = Path(__file__).parent / "models" / "anomaly-detection" / "output"
        
        # Try to load isolation_forest model (best for anomaly detection)
        model_paths = [
            output_dir / "isolation_forest_model.joblib",
            output_dir / "lof_model.joblib",
            models_dir.parent / "output" / "isolation_forest_model.joblib",
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                _anomaly_model = joblib.load(model_path)
                logger.info(f"[AnomalyAPI] Loaded model from {model_path}")
                break
        
        if _anomaly_model is None:
            logger.warning("[AnomalyAPI] No trained model found. Run training first.")
            return False
        
        # Load vectorizer and language detector
        from models.anomaly_detection.src.utils.vectorizer import get_vectorizer
        from models.anomaly_detection.src.utils.language_detector import detect_language
        
        _vectorizer = get_vectorizer()
        _language_detector = detect_language
        
        logger.info("[AnomalyAPI] ✓ All anomaly components loaded")
        return True
        
    except Exception as e:
        logger.error(f"[AnomalyAPI] Failed to load components: {e}")
        return False


@app.post("/api/predict")
def predict_anomaly(texts: List[str] = None, text: str = None):
    """
    Run anomaly detection on text(s).
    
    Args:
        texts: List of texts to analyze
        text: Single text to analyze (alternative to texts)
    
    Returns:
        Predictions with anomaly scores
    """
    try:
        # Handle input
        if text and not texts:
            texts = [text]
        
        if not texts:
            return {"error": "No text provided. Use 'text' or 'texts' field.", "predictions": []}
        
        # Load components
        if not _load_anomaly_components():
            # If no model, return scores based on heuristics
            return {
                "predictions": [
                    {
                        "text": t[:100] + "..." if len(t) > 100 else t,
                        "is_anomaly": False,
                        "anomaly_score": 0.0,
                        "method": "heuristic"
                    }
                    for t in texts
                ],
                "model_status": "not_trained",
                "message": "Model not trained yet. Using default scores."
            }
        
        # Vectorize texts
        predictions = []
        for t in texts:
            try:
                # Detect language
                lang, lang_conf = _language_detector(t)
                
                # Vectorize
                vector = _vectorizer.vectorize(t, lang)
                
                # Predict
                # Isolation Forest returns -1 for anomalies, 1 for normal
                prediction = _anomaly_model.predict([vector])[0]
                
                # Get anomaly score (decision_function returns negative for anomalies)
                if hasattr(_anomaly_model, 'decision_function'):
                    score = -_anomaly_model.decision_function([vector])[0]  # Invert so higher = more anomalous
                elif hasattr(_anomaly_model, 'score_samples'):
                    score = -_anomaly_model.score_samples([vector])[0]
                else:
                    score = 1.0 if prediction == -1 else 0.0
                
                predictions.append({
                    "text": t[:100] + "..." if len(t) > 100 else t,
                    "is_anomaly": prediction == -1,
                    "anomaly_score": float(score),
                    "language": lang,
                    "method": "isolation_forest"
                })
                
            except Exception as e:
                logger.error(f"[AnomalyAPI] Error predicting: {e}")
                predictions.append({
                    "text": t[:100] + "..." if len(t) > 100 else t,
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "error": str(e)
                })
        
        return {
            "predictions": predictions,
            "total": len(predictions),
            "anomalies_found": sum(1 for p in predictions if p.get("is_anomaly")),
            "model_status": "loaded"
        }
        
    except Exception as e:
        logger.error(f"[AnomalyAPI] Predict error: {e}", exc_info=True)
        return {"error": str(e), "predictions": []}


@app.get("/api/anomalies")
def get_anomalies(limit: int = 20, threshold: float = 0.5):
    """
    Get recent feeds that are flagged as anomalies.
    
    Args:
        limit: Max number of results
        threshold: Anomaly score threshold (0-1)
    
    Returns:
        List of anomalous events
    """
    try:
        # Get recent feeds
        feeds = storage_manager.get_recent_feeds(limit=100)
        
        if not feeds:
            # No feeds yet - return helpful message
            return {
                "anomalies": [],
                "total": 0,
                "model_status": "no_data",
                "message": "No feed data available yet. Wait for graph execution to complete."
            }
        
        if not _load_anomaly_components():
            # Use severity + keyword-based scoring as intelligent fallback
            anomalies = []
            anomaly_keywords = ["emergency", "crisis", "breaking", "urgent", "alert", 
                               "warning", "critical", "disaster", "flood", "protest"]
            
            for f in feeds:
                score = 0.0
                summary = str(f.get("summary", "")).lower()
                severity = f.get("severity", "low")
                
                # Severity-based scoring
                if severity == "critical": score = 0.9
                elif severity == "high": score = 0.75
                elif severity == "medium": score = 0.5
                else: score = 0.25
                
                # Keyword boosting
                keyword_matches = sum(1 for kw in anomaly_keywords if kw in summary)
                if keyword_matches > 0:
                    score = min(1.0, score + (keyword_matches * 0.1))
                
                # Only include if above threshold
                if score >= threshold:
                    anomalies.append({
                        **f,
                        "anomaly_score": round(score, 3),
                        "is_anomaly": score >= 0.7
                    })
            
            # Sort by anomaly score
            anomalies.sort(key=lambda x: x.get("anomaly_score", 0), reverse=True)
            
            return {
                "anomalies": anomalies[:limit],
                "total": len(anomalies),
                "threshold": threshold,
                "model_status": "fallback_scoring",
                "message": "Using severity + keyword scoring. Train ML model for advanced detection."
            }
        
        # ML Model is loaded - use it for scoring
        anomalies = []
        for feed in feeds:
            summary = feed.get("summary", "")
            if not summary:
                continue
            
            try:
                lang, _ = _language_detector(summary)
                vector = _vectorizer.vectorize(summary, lang)
                prediction = _anomaly_model.predict([vector])[0]
                
                if hasattr(_anomaly_model, 'decision_function'):
                    score = -_anomaly_model.decision_function([vector])[0]
                else:
                    score = 1.0 if prediction == -1 else 0.0
                
                # Normalize score to 0-1 range
                normalized_score = max(0, min(1, (score + 0.5)))
                
                if prediction == -1 or normalized_score >= threshold:
                    anomalies.append({
                        **feed,
                        "anomaly_score": float(round(normalized_score, 3)),
                        "is_anomaly": prediction == -1,
                        "language": lang
                    })
                    
                    if len(anomalies) >= limit:
                        break
                        
            except Exception as e:
                logger.debug(f"[AnomalyAPI] Error scoring feed: {e}")
                continue
        
        # Sort by anomaly score
        anomalies.sort(key=lambda x: x.get("anomaly_score", 0), reverse=True)
        
        return {
            "anomalies": anomalies,
            "total": len(anomalies),
            "threshold": threshold,
            "model_status": "ml_active"
        }
        
    except Exception as e:
        logger.error(f"[AnomalyAPI] Get anomalies error: {e}")
        return {"anomalies": [], "total": 0, "error": str(e)}


@app.get("/api/model/status")
def get_model_status():
    """Get anomaly detection model status"""
    try:
        from pathlib import Path
        
        output_dir = Path(__file__).parent / "models" / "anomaly-detection" / "output"
        models_found = []
        
        if output_dir.exists():
            for f in output_dir.glob("*.joblib"):
                models_found.append(f.name)
        
        loaded = _anomaly_model is not None
        
        return {
            "model_loaded": loaded,
            "models_available": models_found,
            "vectorizer_loaded": _vectorizer is not None,
            "batch_threshold": int(os.getenv("BATCH_THRESHOLD", "1000")),
            "output_directory": str(output_dir)
        }
        
    except Exception as e:
        return {"error": str(e), "model_loaded": False}


# ============================================
# RAG CHATBOT ENDPOINTS
# ============================================

# Lazy-loaded RAG instance
_rag_instance = None


def _get_rag():
    """Get or create RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        try:
            from src.rag import RogerRAG
            _rag_instance = RogerRAG()
            logger.info("[RAG API] ✓ RAG instance initialized")
        except Exception as e:
            logger.error(f"[RAG API] Failed to initialize RAG: {e}")
            return None
    return _rag_instance


from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    domain_filter: Optional[str] = None
    use_history: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    reformulated: Optional[str] = None
    docs_found: int = 0
    error: Optional[str] = None


@app.post("/api/rag/chat", response_model=ChatResponse)
def rag_chat(request: ChatRequest):
    """
    Chat with the RAG system.
    
    Args:
        message: User's question
        domain_filter: Optional domain (political, economic, weather, social, intelligence)
        use_history: Whether to use chat history for context (default: True)
    
    Returns:
        AI response with sources
    """
    try:
        rag = _get_rag()
        if not rag:
            return ChatResponse(
                answer="RAG system not available. Please check server logs.",
                error="RAG initialization failed"
            )
        
        result = rag.query(
            question=request.message,
            domain_filter=request.domain_filter,
            use_history=request.use_history
        )
        
        return ChatResponse(
            answer=result.get("answer", "No response generated."),
            sources=result.get("sources", []),
            reformulated=result.get("reformulated"),
            docs_found=result.get("docs_found", 0),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"[RAG API] Chat error: {e}", exc_info=True)
        return ChatResponse(
            answer=f"Error processing your request: {str(e)}",
            error=str(e)
        )


@app.get("/api/rag/stats")
def rag_stats():
    """Get RAG system statistics"""
    try:
        rag = _get_rag()
        if not rag:
            return {"error": "RAG not available", "status": "offline"}
        
        stats = rag.get_stats()
        stats["status"] = "online"
        return stats
        
    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.post("/api/rag/clear")
def rag_clear_history():
    """Clear RAG chat history"""
    try:
        rag = _get_rag()
        if rag:
            rag.clear_history()
            return {"message": "Chat history cleared", "success": True}
        return {"message": "RAG not available", "success": False}
        
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# INTELLIGENCE CONFIG ENDPOINTS (User-defined monitoring targets)
# =============================================================================

INTEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "src", "config", "intel_config.json")


def _ensure_intel_config() -> str:
    """Ensure config directory and file exist with default structure"""
    os.makedirs(os.path.dirname(INTEL_CONFIG_PATH), exist_ok=True)
    if not os.path.exists(INTEL_CONFIG_PATH):
        default_config = {
            "user_profiles": {"twitter": [], "facebook": [], "linkedin": []},
            "user_keywords": [],
            "user_products": []
        }
        with open(INTEL_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"[IntelConfig] Created default config at {INTEL_CONFIG_PATH}")
    return INTEL_CONFIG_PATH


@app.get("/api/intel/config")
def get_intel_config():
    """
    Get current intelligence monitoring configuration.
    
    Returns user-defined profiles, keywords, and products that the
    Intelligence Agent monitors in addition to defaults.
    """
    try:
        path = _ensure_intel_config()
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return {"status": "success", "config": config}
    except Exception as e:
        logger.error(f"[IntelConfig] Error reading config: {e}")
        return {"status": "error", "error": str(e)}


class IntelConfigUpdate(BaseModel):
    user_profiles: Optional[Dict[str, List[str]]] = None
    user_keywords: Optional[List[str]] = None
    user_products: Optional[List[str]] = None


@app.post("/api/intel/config")
def update_intel_config(config: IntelConfigUpdate):
    """
    Update intelligence monitoring configuration.
    
    Replaces the entire user config with the provided values.
    """
    try:
        path = _ensure_intel_config()
        
        # Read existing config
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        
        # Update with provided values
        if config.user_profiles is not None:
            existing["user_profiles"] = config.user_profiles
        if config.user_keywords is not None:
            existing["user_keywords"] = config.user_keywords
        if config.user_products is not None:
            existing["user_products"] = config.user_products
        
        # Save
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        
        logger.info(f"[IntelConfig] Updated config: {len(existing.get('user_keywords', []))} keywords, {sum(len(v) for v in existing.get('user_profiles', {}).values())} profiles")
        return {"status": "updated", "config": existing}
        
    except Exception as e:
        logger.error(f"[IntelConfig] Error updating config: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/api/intel/config/add")
def add_intel_target(target_type: str, value: str, platform: Optional[str] = None):
    """
    Add a single monitoring target.
    
    Args:
        target_type: "keyword", "product", or "profile"
        value: The value to add
        platform: Required for "profile" type (twitter, facebook, linkedin)
    
    Example:
        POST /api/intel/config/add?target_type=keyword&value=Colombo+Port
        POST /api/intel/config/add?target_type=profile&value=CompetitorX&platform=twitter
    """
    try:
        path = _ensure_intel_config()
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        added = False
        
        if target_type == "keyword":
            if value not in config.get("user_keywords", []):
                config.setdefault("user_keywords", []).append(value)
                added = True
        elif target_type == "product":
            if value not in config.get("user_products", []):
                config.setdefault("user_products", []).append(value)
                added = True
        elif target_type == "profile":
            if not platform:
                return {"status": "error", "error": "platform is required for profile type"}
            profiles = config.setdefault("user_profiles", {})
            platform_list = profiles.setdefault(platform, [])
            if value not in platform_list:
                platform_list.append(value)
                added = True
        else:
            return {"status": "error", "error": f"Invalid target_type: {target_type}"}
        
        if added:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"[IntelConfig] Added {target_type}: {value}")
        
        return {"status": "added" if added else "already_exists", "config": config}
        
    except Exception as e:
        logger.error(f"[IntelConfig] Error adding target: {e}")
        return {"status": "error", "error": str(e)}


@app.delete("/api/intel/config/remove")
def remove_intel_target(target_type: str, value: str, platform: Optional[str] = None):
    """
    Remove a monitoring target.
    
    Args:
        target_type: "keyword", "product", or "profile"
        value: The value to remove
        platform: Required for "profile" type
    """
    try:
        path = _ensure_intel_config()
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        removed = False
        
        if target_type == "keyword":
            if value in config.get("user_keywords", []):
                config["user_keywords"].remove(value)
                removed = True
        elif target_type == "product":
            if value in config.get("user_products", []):
                config["user_products"].remove(value)
                removed = True
        elif target_type == "profile":
            if not platform:
                return {"status": "error", "error": "platform is required for profile type"}
            if platform in config.get("user_profiles", {}) and value in config["user_profiles"][platform]:
                config["user_profiles"][platform].remove(value)
                removed = True
        else:
            return {"status": "error", "error": f"Invalid target_type: {target_type}"}
        
        if removed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"[IntelConfig] Removed {target_type}: {value}")
        
        return {"status": "removed" if removed else "not_found", "config": config}
        
    except Exception as e:
        logger.error(f"[IntelConfig] Error removing target: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# WEATHER PREDICTION ENDPOINTS
# =============================================================================

# Lazy-loaded weather predictor
_weather_predictor = None

def get_weather_predictor():
    """Lazy-load the weather predictor using isolated import."""
    global _weather_predictor
    if _weather_predictor is not None:
        return _weather_predictor
    
    try:
        import importlib.util
        from pathlib import Path
        
        # Use importlib.util for fully isolated import (avoids package collisions)
        weather_src = Path(__file__).parent / "models" / "weather-prediction" / "src"
        predictor_path = weather_src / "components" / "predictor.py"
        
        if not predictor_path.exists():
            logger.error(f"[WeatherAPI] predictor.py not found at {predictor_path}")
            return None
        
        # First, ensure entity module is loadable
        entity_path = weather_src / "entity" / "config_entity.py"
        if entity_path.exists():
            entity_spec = importlib.util.spec_from_file_location(
                "weather_config_entity",
                str(entity_path)
            )
            entity_module = importlib.util.module_from_spec(entity_spec)
            sys.modules["weather_config_entity"] = entity_module
            entity_spec.loader.exec_module(entity_module)
        
        # Add weather src to path temporarily for relative imports
        import sys
        weather_src_str = str(weather_src)
        if weather_src_str not in sys.path:
            sys.path.insert(0, weather_src_str)
        
        # Now load predictor module
        spec = importlib.util.spec_from_file_location(
            "weather_predictor_module",
            str(predictor_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        _weather_predictor = module.WeatherPredictor()
        logger.info("[WeatherAPI] ✓ Weather predictor initialized via isolated import")
        return _weather_predictor
        
    except Exception as e:
        logger.error(f"[WeatherAPI] Failed to initialize predictor: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


@app.get("/api/weather/predictions")
async def get_weather_predictions():
    """
    Get weather predictions for all 25 Sri Lankan districts.
    
    Returns next-day predictions including:
    - Temperature (high/low)
    - Rainfall (amount and probability)
    - Flood risk
    - Severity classification
    """
    predictor = get_weather_predictor()
    
    if predictor is None:
        return {
            "status": "unavailable",
            "message": "Weather prediction model not loaded",
            "predictions": None
        }
    
    try:
        # Try to get latest predictions from file
        predictions = predictor.get_latest_predictions()
        
        if predictions is None:
            # Generate new predictions
            logger.info("[WeatherAPI] Generating new predictions...")
            predictions = predictor.predict_all_districts()
            predictor.save_predictions(predictions)
        
        return {
            "status": "success",
            "prediction_date": predictions.get("prediction_date"),
            "generated_at": predictions.get("generated_at"),
            "districts": predictions.get("districts", {}),
            "total_districts": len(predictions.get("districts", {}))
        }
    except Exception as e:
        logger.error(f"[WeatherAPI] Error getting predictions: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/weather/predictions/{district}")
async def get_district_weather(district: str):
    """Get weather prediction for a specific district."""
    predictor = get_weather_predictor()
    
    if predictor is None:
        return {"status": "unavailable", "message": "Weather predictor not loaded"}
    
    try:
        predictions = predictor.get_latest_predictions()
        
        if predictions is None:
            predictions = predictor.predict_all_districts()
        
        districts = predictions.get("districts", {})
        
        # Case-insensitive lookup
        district_key = None
        for d in districts.keys():
            if d.lower() == district.lower():
                district_key = d
                break
        
        if district_key is None:
            return {
                "status": "not_found",
                "message": f"District '{district}' not found",
                "available_districts": list(districts.keys())
            }
        
        return {
            "status": "success",
            "district": district_key,
            "prediction_date": predictions.get("prediction_date"),
            "prediction": districts[district_key]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/weather/model/status")
async def get_weather_model_status():
    """Get weather prediction model status and training info."""
    from pathlib import Path
    import os
    
    models_dir = Path(__file__).parent / "models" / "weather-prediction" / "artifacts" / "models"
    predictions_dir = Path(__file__).parent / "models" / "weather-prediction" / "output" / "predictions"
    
    model_files = list(models_dir.glob("lstm_*.h5")) if models_dir.exists() else []
    prediction_files = list(predictions_dir.glob("predictions_*.json")) if predictions_dir.exists() else []
    
    latest_prediction = None
    if prediction_files:
        latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
        latest_prediction = {
            "file": latest.name,
            "modified": datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
        }
    
    return {
        "status": "available" if model_files else "not_trained",
        "models_trained": len(model_files),
        "trained_stations": [f.stem.replace("lstm_", "").upper() for f in model_files],
        "latest_prediction": latest_prediction,
        "predictions_available": len(prediction_files)
    }


# =============================================================================
# CURRENCY PREDICTION ENDPOINTS
# =============================================================================

# Lazy-loaded currency predictor
_currency_predictor = None

def get_currency_predictor():
    """Lazy-load the currency predictor."""
    global _currency_predictor
    if _currency_predictor is None:
        try:
            import sys
            from pathlib import Path
            currency_path = Path(__file__).parent / "models" / "currency-volatility-prediction" / "src"
            sys.path.insert(0, str(currency_path))
            from components.predictor import CurrencyPredictor
            _currency_predictor = CurrencyPredictor()
            logger.info("[CurrencyAPI] Currency predictor initialized")
        except Exception as e:
            logger.warning(f"[CurrencyAPI] Failed to initialize predictor: {e}")
            _currency_predictor = None
    return _currency_predictor


@app.get("/api/currency/prediction")
async def get_currency_prediction():
    """
    Get USD/LKR currency prediction for next day.
    
    Returns:
    - Current rate
    - Predicted rate
    - Expected change percentage
    - Direction (strengthening/weakening)
    - Volatility classification
    """
    predictor = get_currency_predictor()
    
    if predictor is None:
        return {
            "status": "unavailable",
            "message": "Currency prediction model not loaded"
        }
    
    try:
        # Try to get latest prediction from file
        prediction = predictor.get_latest_prediction()
        
        if prediction is None:
            # Generate fallback
            logger.info("[CurrencyAPI] No prediction found, generating fallback...")
            prediction = predictor.generate_fallback_prediction()
            predictor.save_prediction(prediction)
        
        return {
            "status": "success",
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"[CurrencyAPI] Error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/currency/history")
async def get_currency_history(days: int = 30):
    """Get historical USD/LKR rates."""
    from pathlib import Path
    import pandas as pd
    
    try:
        data_dir = Path(__file__).parent / "models" / "currency-volatility-prediction" / "artifacts" / "data"
        csv_files = list(data_dir.glob("currency_data_*.csv")) if data_dir.exists() else []
        
        if not csv_files:
            return {"status": "no_data", "message": "No currency data available"}
        
        latest = max(csv_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest, parse_dates=["date"])
        
        # Get last N days
        df = df.tail(days)
        
        history = []
        for _, row in df.iterrows():
            history.append({
                "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                "close": round(row["close"], 2),
                "high": round(row.get("high", row["close"]), 2),
                "low": round(row.get("low", row["close"]), 2),
                "daily_return_pct": round(row.get("daily_return", 0) * 100, 3)
            })
        
        return {
            "status": "success",
            "days": len(history),
            "history": history
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/currency/model/status")
async def get_currency_model_status():
    """Get currency prediction model status."""
    from pathlib import Path
    
    models_dir = Path(__file__).parent / "models" / "currency-volatility-prediction" / "artifacts" / "models"
    predictions_dir = Path(__file__).parent / "models" / "currency-volatility-prediction" / "output" / "predictions"
    
    model_exists = (models_dir / "gru_usd_lkr.h5").exists() if models_dir.exists() else False
    prediction_files = list(predictions_dir.glob("currency_prediction_*.json")) if predictions_dir.exists() else []
    
    latest_prediction = None
    if prediction_files:
        latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
        latest_prediction = {
            "file": latest.name,
            "modified": datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
        }
    
    return {
        "status": "available" if model_exists else "not_trained",
        "model_type": "GRU",
        "target": "USD/LKR",
        "latest_prediction": latest_prediction,
        "predictions_available": len(prediction_files)
    }


# =============================================================================
# STOCK PREDICTION ENDPOINTS
# =============================================================================

# Lazy-loaded stock predictor
_stock_predictor = None

def get_stock_predictor():
    """Lazy-load the stock predictor."""
    global _stock_predictor
    if _stock_predictor is None:
        try:
            import sys
            from pathlib import Path
            stock_path = Path(__file__).parent / "models" / "stock-price-prediction" / "src"
            sys.path.insert(0, str(stock_path))
            from components.predictor import StockPredictor
            _stock_predictor = StockPredictor()
            logger.info("[StockAPI] Stock predictor initialized")
        except Exception as e:
            logger.warning(f"[StockAPI] Failed to initialize predictor: {e}")
            _stock_predictor = None
    return _stock_predictor


@app.get("/api/stocks/predictions")
async def get_stock_predictions():
    """
    Get stock price predictions for all CSE stocks.
    
    Returns predictions for 10 top Sri Lankan stocks with:
    - Current price
    - Predicted next-day price
    - Expected change percentage
    - Trend classification (bullish/bearish/neutral)
    - Model architecture used
    """
    predictor = get_stock_predictor()
    
    if predictor is None:
        return {
            "status": "unavailable",
            "message": "Stock prediction model not loaded"
        }
    
    try:
        # Try to get latest predictions from file
        predictions = predictor.get_latest_predictions()
        
        if predictions is None:
            # Generate fallback predictions
            logger.info("[StockAPI] No predictions found, generating fallback...")
            from entity.config_entity import SRI_LANKA_STOCKS
            predictions = {
                "prediction_date": (datetime.now()).strftime("%Y-%m-%d"),
                "generated_at": datetime.now().isoformat(),
                "stocks": {
                    code: predictor._generate_fallback_prediction(code)
                    for code in SRI_LANKA_STOCKS.keys()
                },
                "summary": {"total_stocks": len(SRI_LANKA_STOCKS)}
            }
        
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"[StockAPI] Error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/stocks/predictions/{symbol}")
async def get_stock_prediction_by_symbol(symbol: str):
    """Get prediction for a specific stock symbol."""
    predictor = get_stock_predictor()
    
    if predictor is None:
        return {"status": "unavailable", "message": "Stock prediction model not loaded"}
    
    try:
        predictions = predictor.get_latest_predictions()
        
        if predictions and symbol.upper() in predictions.get("stocks", {}):
            return {
                "status": "success",
                "prediction": predictions["stocks"][symbol.upper()]
            }
        else:
            # Generate fallback
            return {
                "status": "success",
                "prediction": predictor._generate_fallback_prediction(symbol.upper())
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/stocks/model/status")
async def get_stock_model_status():
    """Get stock prediction model status for all stocks."""
    from pathlib import Path
    import json
    
    models_dir = Path(__file__).parent / "models" / "stock-price-prediction" / "artifacts" / "models"
    predictions_dir = Path(__file__).parent / "models" / "stock-price-prediction" / "output" / "predictions"
    
    model_files = list(models_dir.glob("*_model.h5")) if models_dir.exists() else []
    prediction_files = list(predictions_dir.glob("stock_predictions_*.json")) if predictions_dir.exists() else []
    
    # Get training summary
    summary_path = models_dir / "training_summary.json" if models_dir.exists() else None
    training_summary = None
    if summary_path and summary_path.exists():
        with open(summary_path) as f:
            training_summary = json.load(f)
    
    latest_prediction = None
    if prediction_files:
        latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
        latest_prediction = {
            "file": latest.name,
            "modified": datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
        }
    
    return {
        "status": "available" if model_files else "not_trained",
        "models_trained": len(model_files),
        "trained_stocks": [f.stem.replace("_model", "").upper() for f in model_files],
        "training_summary": training_summary,
        "latest_prediction": latest_prediction,
        "predictions_available": len(prediction_files)
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial state
        try:
            await websocket.send_text(json.dumps(current_state, default=str))
        except Exception as e:
            logger.debug(f"[WS] Initial send failed: {e}")
            await manager.disconnect(websocket)
            return

        # Main receive loop
        while True:
            try:
                txt = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("[WS] Client disconnected")
                break
            except Exception as e:
                logger.debug(f"[WS] Receive error: {e}")
                break

            # Handle pong responses
            try:
                payload = json.loads(txt)
                if isinstance(payload, dict) and payload.get("type") == "pong":
                    async with manager._lock:
                        meta = manager.active_connections.get(websocket)
                        if meta is not None:
                            meta['last_pong'] = datetime.utcnow()
                            meta['misses'] = 0
                    continue
            except json.JSONDecodeError:
                continue

    finally:
        await manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    import uuid
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
