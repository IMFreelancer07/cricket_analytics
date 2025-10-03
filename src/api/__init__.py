"""
FastAPI Application for Cricket Analytics RAG System
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
from contextlib import asynccontextmanager

# Import our cricket modules (with error handling for development)
try:
    from ..rag import CricketRAG, setup_cricket_rag
    from ..langgraph import CricketAnalyzer, create_cricket_analyzer
except ImportError:
    CricketRAG = None
    CricketAnalyzer = None
    setup_cricket_rag = None
    create_cricket_analyzer = None
    print("Warning: Cricket modules not available. Running in development mode.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
cricket_rag = None
cricket_analyzer = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    """Manage startup and shutdown lifecycle for the API."""
    global cricket_rag, cricket_analyzer

    try:
        logger.info("Initializing Cricket Analytics services...")

        if CricketRAG and setup_cricket_rag:
            cricket_rag = setup_cricket_rag()
            logger.info("Cricket RAG system initialized")
        else:
            logger.warning("Cricket RAG system not available")

        if CricketAnalyzer and create_cricket_analyzer:
            cricket_analyzer = create_cricket_analyzer(rag_system=cricket_rag)
            logger.info("Cricket LangGraph analyzer initialized")
        else:
            logger.warning("Cricket LangGraph analyzer not available")

        logger.info("Cricket Analytics API startup completed")
    except Exception as exc:
        logger.error(f"Failed to initialize services: {exc}")

    try:
        yield
    finally:
        logger.info("Shutting down Cricket Analytics API")
        cricket_analyzer = None
        cricket_rag = None


# FastAPI app
app = FastAPI(
    title="Cricket Analytics RAG API",
    description="Agentic AI system for cricket match analytics using RAG + LangChain",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Cricket analytics query")
    query_type: str = Field(default="general", description="Type of query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")
    use_complex_reasoning: bool = Field(default=False, description="Use multi-hop reasoning")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    query_type: str
    num_sources: int
    reasoning_steps: Optional[List[str]] = None
    processing_time_ms: Optional[float] = None
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Dependency functions
def get_cricket_rag():
    """Dependency to get cricket RAG system"""
    if cricket_rag is None:
        raise HTTPException(
            status_code=503, 
            detail="Cricket RAG system not available"
        )
    return cricket_rag

def get_cricket_analyzer():
    """Dependency to get cricket analyzer"""
    if cricket_analyzer is None:
        raise HTTPException(
            status_code=503, 
            detail="Cricket analyzer not available"
        )
    return cricket_analyzer

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Cricket Analytics RAG API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {
        "rag_system": "available" if cricket_rag else "unavailable",
        "langgraph_analyzer": "available" if cricket_analyzer else "unavailable",
        "database": "not_implemented",
        "vector_store": "available" if cricket_rag and cricket_rag.vectorstore else "unavailable"
    }
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services=services,
        timestamp=datetime.now().isoformat()
    )

@app.post("/query", response_model=QueryResponse)
async def query_cricket_analytics(
    request: QueryRequest,
    rag_system = Depends(get_cricket_rag)
):
    """
    Query cricket analytics using RAG system
    
    - **query**: Natural language query about cricket
    - **query_type**: Type of query (general, player_stats, match_analysis, tactical, comparison)
    - **filters**: Optional filters for specific matches, players, venues
    - **use_complex_reasoning**: Whether to use multi-hop reasoning via LangGraph
    """
    start_time = datetime.now()
    
    try:
        if request.use_complex_reasoning and cricket_analyzer:
            # Use complex reasoning via LangGraph
            result = cricket_analyzer.analyze_complex_query(request.query)
            
            if result.get("status") == "success":
                response = QueryResponse(
                    query=request.query,
                    answer=result.get("final_answer", "No answer generated"),
                    sources=[],  # LangGraph provides different source structure
                    query_type=request.query_type,
                    num_sources=0,
                    reasoning_steps=cricket_analyzer.visualize_reasoning_path(request.query)["reasoning_steps"],
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    timestamp=datetime.now().isoformat()
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Complex reasoning failed: {result.get('error', 'Unknown error')}"
                )
        else:
            # Use standard RAG query
            if request.filters:
                result = rag_system.query_with_filters(request.query, request.filters)
            else:
                result = rag_system.query(request.query, request.query_type)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            response = QueryResponse(
                query=request.query,
                answer=result["answer"],
                sources=result.get("sources", []),
                query_type=request.query_type,
                num_sources=result.get("num_sources", 0),
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat()
            )
        
        logger.info(f"Successfully processed query: {request.query[:50]}...")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complex-analysis", response_model=QueryResponse)
async def complex_cricket_analysis(
    request: QueryRequest,
    analyzer = Depends(get_cricket_analyzer)
):
    """
    Perform complex cricket analysis using multi-hop reasoning
    
    This endpoint specifically uses LangGraph for sophisticated multi-step analysis
    """
    start_time = datetime.now()
    
    try:
        result = analyzer.analyze_complex_query(request.query)
        
        if result.get("status") == "success":
            reasoning_path = analyzer.visualize_reasoning_path(request.query)
            
            response = QueryResponse(
                query=request.query,
                answer=result.get("final_answer", "No answer generated"),
                sources=[],  # Complex analysis doesn't use traditional sources
                query_type="complex_analysis",
                num_sources=0,
                reasoning_steps=reasoning_path["reasoning_steps"],
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Successfully completed complex analysis: {request.query[:50]}...")
            return response
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Complex analysis failed: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform complex analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples", response_model=List[Dict[str, str]])
async def get_example_queries():
    """Get example cricket analytics queries"""
    return [
        {
            "query": "How does Virat Kohli perform against left-arm pacers in the death overs?",
            "type": "player_analysis",
            "description": "Player-specific performance analysis with situational context"
        },
        {
            "query": "What bowling changes worked best in the last 5 matches at Eden Gardens?",
            "type": "tactical_analysis",
            "description": "Venue-specific tactical analysis"
        },
        {
            "query": "Analyze team performance in chase scenarios under 15 overs",
            "type": "match_analysis",
            "description": "Situational team performance analysis"
        },
        {
            "query": "Compare batting averages in powerplay vs middle overs for top 5 batsmen",
            "type": "comparison",
            "description": "Comparative statistical analysis"
        },
        {
            "query": "What tactical changes should be made for the next match based on recent performance?",
            "type": "strategic_recommendation",
            "description": "Complex multi-hop reasoning for strategic insights"
        }
    ]

@app.get("/reasoning-graph", response_model=Dict[str, Any])
async def get_reasoning_graph_info(
    analyzer = Depends(get_cricket_analyzer)
):
    """Get information about the reasoning graph structure"""
    try:
        graph_info = {
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type.value,
                    "description": node.description
                }
                for node in analyzer.reasoning_graph.nodes.values()
            ],
            "edges": [
                {
                    "from": edge.from_node,
                    "to": edge.to_node
                }
                for edge in analyzer.reasoning_graph.edges
            ],
            "reasoning_flow": "data_retrieval → [player_analysis, match_analysis] → tactical_reasoning → synthesis"
        }
        return graph_info
        
    except Exception as e:
        logger.error(f"Failed to get reasoning graph info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return ErrorResponse(
        error=exc.detail,
        detail=f"HTTP {exc.status_code}",
        timestamp=datetime.now().isoformat()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.now().isoformat()
    )

# Additional utility endpoints
@app.get("/stats", response_model=Dict[str, Any])
async def get_api_stats():
    """Get API usage statistics"""
    # Placeholder for actual statistics
    return {
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "average_response_time_ms": 0,
        "popular_query_types": [],
        "uptime": "Not implemented"
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true"
    )