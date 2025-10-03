"""
RAG (Retrieval-Augmented Generation) Module for Cricket Analytics
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain.schema import Document
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class CricketQuery:
    """Structured cricket query with metadata"""
    query: str
    query_type: str  # 'player_stats', 'match_analysis', 'tactical', 'comparison'
    context: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None

class CricketEmbeddingManager:
    """Manage embeddings for cricket data"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_embeddings(self, documents: List[Document]) -> Chroma:
        """
        Create vector embeddings for cricket documents
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Chroma vector store
        """
        try:
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory="./data/vectordb",
                collection_name="cricket_analytics"
            )
            
            vectorstore.persist()
            logger.info(f"Created embeddings for {len(split_docs)} document chunks")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return None
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store if available"""
        try:
            vectorstore = Chroma(
                persist_directory="./data/vectordb",
                embedding_function=self.embeddings,
                collection_name="cricket_analytics"
            )
            logger.info("Loaded existing vector store")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to load existing vector store: {e}")
            return None

class CricketRetriever:
    """Retrieve relevant cricket documents based on queries"""
    
    def __init__(self, vectorstore: Chroma, k: int = 5):
        self.vectorstore = vectorstore
        self.k = k
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    def retrieve_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve relevant documents for a cricket query
        
        Args:
            query: Search query
            filters: Optional filters for document metadata
            
        Returns:
            List of relevant documents
        """
        try:
            # Apply filters if provided
            if filters:
                search_kwargs = {"k": self.k, "filter": filters}
                retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            else:
                retriever = self.retriever
            
            documents = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Retrieve documents with similarity scores"""
        try:
            k = k or self.k
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve documents with scores: {e}")
            return []

class CricketRAG:
    """Main RAG system for cricket analytics"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.3):
        
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm: Optional[OpenAI] = None
        if self.api_key:
            self.llm = OpenAI(
                openai_api_key=self.api_key,
                model_name=model_name,
                temperature=temperature
            )
        else:
            logger.warning("OpenAI API key not provided. LLM-powered features will be disabled.")
        
        self.embedding_manager = CricketEmbeddingManager()
        self.vectorstore = self.embedding_manager.load_existing_vectorstore()
        
        if self.vectorstore:
            self.retriever = CricketRetriever(self.vectorstore)
        else:
            self.retriever = None
            logger.warning("No vector store available. Call initialize_vectorstore() first.")
        
        self.qa_chain = None
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Setup the QA chain with cricket-specific prompts"""
        cricket_prompt_template = """
        You are an expert cricket analyst with deep knowledge of cricket strategy, player performance, and match dynamics.
        
        Use the following cricket data to answer the question. Focus on providing tactical insights, statistical analysis, and strategic recommendations.
        
        Context from cricket data:
        {context}
        
        Question: {question}
        
        Provide a comprehensive answer that includes:
        1. Direct answer to the question
        2. Supporting statistics and evidence
        3. Tactical insights and strategic implications
        4. Historical context if relevant
        
        Answer:
        """
        
        cricket_prompt = PromptTemplate(
            template=cricket_prompt_template,
            input_variables=["context", "question"]
        )
        
        if self.llm and self.retriever and self.vectorstore:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever.retriever,
                chain_type_kwargs={"prompt": cricket_prompt},
                return_source_documents=True
            )
    
    def initialize_vectorstore(self, data_directory: str = "data/processed") -> None:
        """Initialize vector store with cricket data"""
        try:
            documents = self._load_cricket_documents(data_directory)
            
            if documents:
                self.vectorstore = self.embedding_manager.create_embeddings(documents)
                self.retriever = CricketRetriever(self.vectorstore)
                self._setup_qa_chain()
                logger.info("Vector store initialized successfully")
            else:
                logger.warning("No documents found to initialize vector store")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    def _load_cricket_documents(self, data_directory: str) -> List[Document]:
        """Load cricket documents from various sources"""
        documents = []
        data_path = Path(data_directory)
        
        try:
            # Load JSON files (commentary, reports)
            for json_file in data_path.glob("*.json"):
                try:
                    loader = JSONLoader(
                        file_path=str(json_file),
                        jq_schema=".[]",
                        text_content=False
                    )
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
            
            # Load CSV files (stats, performance data)
            for csv_file in data_path.glob("*.csv"):
                try:
                    loader = CSVLoader(file_path=str(csv_file))
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
            
            logger.info(f"Loaded {len(documents)} documents from {data_directory}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load cricket documents: {e}")
            return []
    
    def query(self, cricket_query: str, query_type: str = "general") -> Dict[str, Any]:
        """
        Query the cricket RAG system
        
        Args:
            cricket_query: Natural language query about cricket
            query_type: Type of query for specialized handling
            
        Returns:
            Dictionary containing answer and source information
        """
        if not self.qa_chain:
            return {
                "answer": "RAG system not initialized. Please call initialize_vectorstore() first.",
                "sources": [],
                "error": "System not ready"
            }
        
        try:
            # Process query through QA chain
            result = self.qa_chain({"query": cricket_query})
            
            response = {
                "answer": result["result"],
                "sources": [doc.page_content[:200] + "..." for doc in result.get("source_documents", [])],
                "query_type": query_type,
                "num_sources": len(result.get("source_documents", []))
            }
            
            logger.info(f"Successfully processed query: {cricket_query[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your cricket query.",
                "sources": [],
                "error": str(e)
            }
    
    def query_with_filters(
        self,
        cricket_query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query with metadata filters (e.g., specific matches, players, venues)"""
        
        if not self.retriever:
            return self.query(cricket_query)
        
        try:
            # Get filtered documents
            documents = self.retriever.retrieve_documents(cricket_query, filters)
            
            if not documents:
                return {
                    "answer": "No relevant cricket data found for your query with the specified filters.",
                    "sources": [],
                    "filters_applied": filters
                }
            
            # Create context from filtered documents
            context = "\n\n".join([doc.page_content for doc in documents])
            
            if not self.llm:
                logger.warning("LLM is not configured; returning context without generated answer.")
                return {
                    "answer": "LLM is not configured. Please provide an OpenAI API key to enable generated insights.",
                    "sources": [doc.page_content[:200] + "..." for doc in documents],
                    "filters_applied": filters,
                    "num_sources": len(documents)
                }

            # Generate answer using LLM
            prompt = f"""
            Based on the following cricket data, answer this question: {cricket_query}
            
            Cricket Data:
            {context}
            
            Provide a detailed analysis with supporting statistics and insights.
            """
            
            answer = self.llm(prompt)
            
            return {
                "answer": answer,
                "sources": [doc.page_content[:200] + "..." for doc in documents],
                "filters_applied": filters,
                "num_sources": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Failed to process filtered query: {e}")
            return self.query(cricket_query)


# Utility functions
def setup_cricket_rag(data_directory: str = "data/processed") -> Optional[CricketRAG]:
    """Setup and initialize cricket RAG system"""
    try:
        rag = CricketRAG()
        rag.initialize_vectorstore(data_directory)
        return rag
    except Exception as e:
        logger.error(f"Failed to setup cricket RAG: {e}")
        return None

def example_queries():
    """Example cricket queries for testing"""
    return [
        "How does Virat Kohli perform against left-arm pacers in the death overs?",
        "What bowling changes worked best in the last 5 matches at Eden Gardens?",
        "Analyze team performance in chase scenarios under 15 overs",
        "Compare batting averages in powerplay vs middle overs",
        "What are the key tactical insights from recent matches?"
    ]


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    rag = setup_cricket_rag()
    
    if rag:
        for query in example_queries()[:2]:  # Test first 2 queries
            print(f"\nQuery: {query}")
            result = rag.query(query)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result.get('sources', []))}")