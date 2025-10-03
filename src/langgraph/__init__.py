"""
LangGraph Module for Multi-hop Reasoning in Cricket Analytics
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of reasoning nodes in the cricket analysis graph"""
    DATA_RETRIEVAL = "data_retrieval"
    PLAYER_ANALYSIS = "player_analysis"
    MATCH_ANALYSIS = "match_analysis"
    TACTICAL_REASONING = "tactical_reasoning"
    COMPARISON = "comparison"
    SYNTHESIS = "synthesis"

@dataclass
class CricketNode:
    """Node in the cricket reasoning graph"""
    node_id: str
    node_type: NodeType
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CricketEdge:
    """Edge connecting nodes in the cricket reasoning graph"""
    from_node: str
    to_node: str
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CricketReasoningGraph:
    """Graph-based reasoning system for complex cricket queries"""
    
    def __init__(self):
        self.nodes: Dict[str, CricketNode] = {}
        self.edges: List[CricketEdge] = []
        self.execution_state: Dict[str, Any] = {}
        
    def add_node(self, node: CricketNode):
        """Add a reasoning node to the graph"""
        self.nodes[node.node_id] = node
        logger.info(f"Added node: {node.node_id} ({node.node_type.value})")
    
    def add_edge(self, edge: CricketEdge):
        """Add an edge between nodes"""
        self.edges.append(edge)
        logger.info(f"Added edge: {edge.from_node} -> {edge.to_node}")
    
    def execute_node(self, node_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific node with given inputs"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in graph")
        
        node = self.nodes[node_id]
        
        try:
            if node.function:
                result = node.function(inputs)
                logger.info(f"Executed node {node_id} successfully")
                return result
            else:
                logger.warning(f"No function defined for node {node_id}")
                return {"status": "no_function", "node_id": node_id}
                
        except Exception as e:
            logger.error(f"Failed to execute node {node_id}: {e}")
            return {"status": "error", "error": str(e), "node_id": node_id}
    
    def find_execution_path(self, start_node: str, end_node: str) -> List[str]:
        """Find execution path between two nodes"""
        # Simple BFS implementation for path finding
        from collections import deque
        
        queue = deque([(start_node, [start_node])])
        visited = {start_node}
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node == end_node:
                return path
            
            # Find connected nodes
            for edge in self.edges:
                if edge.from_node == current_node and edge.to_node not in visited:
                    visited.add(edge.to_node)
                    queue.append((edge.to_node, path + [edge.to_node]))
        
        return []  # No path found

class CricketAnalyzer:
    """Main analyzer using LangGraph for complex cricket reasoning"""
    
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
        self.reasoning_graph = CricketReasoningGraph()
        self._setup_default_graph()
    
    def _setup_default_graph(self):
        """Setup default reasoning graph for cricket analysis"""
        
        # Data Retrieval Node
        retrieval_node = CricketNode(
            node_id="data_retrieval",
            node_type=NodeType.DATA_RETRIEVAL,
            description="Retrieve relevant cricket data",
            function=self._retrieve_cricket_data
        )
        
        # Player Analysis Node
        player_analysis_node = CricketNode(
            node_id="player_analysis",
            node_type=NodeType.PLAYER_ANALYSIS,
            description="Analyze individual player performance",
            function=self._analyze_player_performance
        )
        
        # Match Analysis Node
        match_analysis_node = CricketNode(
            node_id="match_analysis",
            node_type=NodeType.MATCH_ANALYSIS,
            description="Analyze match-level patterns",
            function=self._analyze_match_patterns
        )
        
        # Tactical Reasoning Node
        tactical_node = CricketNode(
            node_id="tactical_reasoning",
            node_type=NodeType.TACTICAL_REASONING,
            description="Generate tactical insights",
            function=self._generate_tactical_insights
        )
        
        # Synthesis Node
        synthesis_node = CricketNode(
            node_id="synthesis",
            node_type=NodeType.SYNTHESIS,
            description="Synthesize final answer",
            function=self._synthesize_answer
        )
        
        # Add nodes to graph
        for node in [retrieval_node, player_analysis_node, match_analysis_node, tactical_node, synthesis_node]:
            self.reasoning_graph.add_node(node)
        
        # Add edges
        edges = [
            CricketEdge("data_retrieval", "player_analysis"),
            CricketEdge("data_retrieval", "match_analysis"),
            CricketEdge("player_analysis", "tactical_reasoning"),
            CricketEdge("match_analysis", "tactical_reasoning"),
            CricketEdge("tactical_reasoning", "synthesis")
        ]
        
        for edge in edges:
            self.reasoning_graph.add_edge(edge)
    
    def _retrieve_cricket_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Node function: Retrieve relevant cricket data"""
        query = inputs.get("query", "")
        
        if self.rag_system:
            try:
                rag_result = self.rag_system.query(query)
                return {
                    "status": "success",
                    "data": rag_result,
                    "retrieved_documents": rag_result.get("sources", []),
                    "query": query
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "query": query
                }
        else:
            # Placeholder data retrieval
            return {
                "status": "success",
                "data": {"answer": f"Retrieved data for: {query}"},
                "retrieved_documents": ["Sample cricket data document"],
                "query": query
            }
    
    def _analyze_player_performance(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Node function: Analyze player performance"""
        try:
            data = inputs.get("data", {})
            query = inputs.get("query", "")
            
            # Extract player-related information
            player_insights = {
                "performance_metrics": {},
                "strengths": [],
                "weaknesses": [],
                "situational_performance": {},
                "trends": []
            }
            
            # Placeholder analysis - in real implementation, use ML models
            if "kohli" in query.lower():
                player_insights.update({
                    "performance_metrics": {
                        "average": 50.2,
                        "strike_rate": 92.8,
                        "centuries": 43
                    },
                    "strengths": ["Chase master", "Consistent performer"],
                    "situational_performance": {
                        "death_overs": {"average": 48.5, "strike_rate": 135.2}
                    }
                })
            
            return {
                "status": "success",
                "player_analysis": player_insights,
                "query": query
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _analyze_match_patterns(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Node function: Analyze match-level patterns"""
        try:
            data = inputs.get("data", {})
            query = inputs.get("query", "")
            
            match_patterns = {
                "venue_analysis": {},
                "conditions_impact": {},
                "team_performance": {},
                "tactical_patterns": []
            }
            
            # Placeholder analysis
            if "eden gardens" in query.lower():
                match_patterns.update({
                    "venue_analysis": {
                        "average_score": 165,
                        "successful_chases": "68%",
                        "toss_impact": "Win toss, win match 58%"
                    },
                    "tactical_patterns": [
                        "Early powerplay aggression works well",
                        "Spinners effective in middle overs"
                    ]
                })
            
            return {
                "status": "success",
                "match_patterns": match_patterns,
                "query": query
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_tactical_insights(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Node function: Generate tactical insights"""
        try:
            player_analysis = inputs.get("player_analysis", {})
            match_patterns = inputs.get("match_patterns", {})
            query = inputs.get("query", "")
            
            tactical_insights = {
                "strategic_recommendations": [],
                "key_factors": [],
                "success_probability": {},
                "alternative_strategies": []
            }
            
            # Combine player and match analysis for tactical insights
            if player_analysis and match_patterns:
                tactical_insights.update({
                    "strategic_recommendations": [
                        "Target specific bowling matchups",
                        "Optimize batting order for conditions",
                        "Plan bowling changes based on venue history"
                    ],
                    "key_factors": [
                        "Player form vs opposition",
                        "Venue-specific strategies",
                        "Historical success patterns"
                    ]
                })
            
            return {
                "status": "success",
                "tactical_insights": tactical_insights,
                "query": query
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _synthesize_answer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Node function: Synthesize final comprehensive answer"""
        try:
            tactical_insights = inputs.get("tactical_insights", {})
            player_analysis = inputs.get("player_analysis", {})
            match_patterns = inputs.get("match_patterns", {})
            query = inputs.get("query", "")
            
            # Synthesize comprehensive answer
            final_answer = f"""
            Cricket Analysis for: {query}
            
            ## Player Performance Analysis
            {self._format_player_analysis(player_analysis)}
            
            ## Match Pattern Analysis
            {self._format_match_patterns(match_patterns)}
            
            ## Tactical Insights & Recommendations
            {self._format_tactical_insights(tactical_insights)}
            
            ## Summary
            Based on the multi-dimensional analysis combining player performance, 
            match patterns, and tactical considerations, the key recommendations are...
            """
            
            return {
                "status": "success",
                "final_answer": final_answer.strip(),
                "components": {
                    "player_analysis": player_analysis,
                    "match_patterns": match_patterns,
                    "tactical_insights": tactical_insights
                },
                "query": query
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _format_player_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format player analysis for final answer"""
        if not analysis:
            return "No player-specific analysis available."
        
        formatted = ""
        if "performance_metrics" in analysis:
            formatted += f"Performance Metrics: {analysis['performance_metrics']}\n"
        if "strengths" in analysis:
            formatted += f"Key Strengths: {', '.join(analysis['strengths'])}\n"
        
        return formatted
    
    def _format_match_patterns(self, patterns: Dict[str, Any]) -> str:
        """Format match patterns for final answer"""
        if not patterns:
            return "No match pattern analysis available."
        
        formatted = ""
        if "venue_analysis" in patterns:
            formatted += f"Venue Analysis: {patterns['venue_analysis']}\n"
        if "tactical_patterns" in patterns:
            formatted += f"Tactical Patterns: {patterns['tactical_patterns']}\n"
        
        return formatted
    
    def _format_tactical_insights(self, insights: Dict[str, Any]) -> str:
        """Format tactical insights for final answer"""
        if not insights:
            return "No tactical insights available."
        
        formatted = ""
        if "strategic_recommendations" in insights:
            formatted += f"Strategic Recommendations:\n"
            for rec in insights["strategic_recommendations"]:
                formatted += f"- {rec}\n"
        
        return formatted
    
    def analyze_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze complex cricket query using multi-hop reasoning
        
        Args:
            query: Complex cricket query requiring multi-step reasoning
            
        Returns:
            Comprehensive analysis result
        """
        try:
            logger.info(f"Starting complex analysis for: {query}")
            
            # Execute reasoning graph
            execution_state = {"query": query}
            
            # Step 1: Data Retrieval
            retrieval_result = self.reasoning_graph.execute_node("data_retrieval", execution_state)
            execution_state.update(retrieval_result)
            
            # Step 2: Parallel Analysis
            player_result = self.reasoning_graph.execute_node("player_analysis", execution_state)
            match_result = self.reasoning_graph.execute_node("match_analysis", execution_state)
            
            execution_state.update(player_result)
            execution_state.update(match_result)
            
            # Step 3: Tactical Reasoning
            tactical_result = self.reasoning_graph.execute_node("tactical_reasoning", execution_state)
            execution_state.update(tactical_result)
            
            # Step 4: Synthesis
            final_result = self.reasoning_graph.execute_node("synthesis", execution_state)
            
            logger.info("Complex analysis completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Failed to analyze complex query: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    def add_custom_reasoning_node(self, node: CricketNode):
        """Add custom reasoning node to the graph"""
        self.reasoning_graph.add_node(node)
    
    def visualize_reasoning_path(self, query: str) -> Dict[str, Any]:
        """Visualize the reasoning path for a query"""
        # This would generate a visual representation of the reasoning graph
        # For now, return a textual representation
        return {
            "query": query,
            "reasoning_steps": [
                "1. Data Retrieval - Gather relevant cricket information",
                "2. Player Analysis - Analyze individual performance patterns",
                "3. Match Analysis - Examine venue and match-level patterns",
                "4. Tactical Reasoning - Generate strategic insights",
                "5. Synthesis - Combine insights into comprehensive answer"
            ],
            "node_types": [node.node_type.value for node in self.reasoning_graph.nodes.values()],
            "execution_flow": "data_retrieval → player_analysis → tactical_reasoning → synthesis"
        }


def create_cricket_analyzer(rag_system=None) -> CricketAnalyzer:
    """Factory function to create cricket analyzer"""
    return CricketAnalyzer(rag_system=rag_system)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    analyzer = create_cricket_analyzer()
    
    test_queries = [
        "How does Virat Kohli perform against left-arm pacers in the death overs?",
        "What bowling changes worked best in the last 5 matches at Eden Gardens?"
    ]
    
    for query in test_queries:
        print(f"\n=== Analysis for: {query} ===")
        result = analyzer.analyze_complex_query(query)
        
        if result.get("status") == "success":
            print(result.get("final_answer", "No answer generated"))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"\nReasoning Path:")
        path_info = analyzer.visualize_reasoning_path(query)
        for step in path_info["reasoning_steps"]:
            print(f"  {step}")