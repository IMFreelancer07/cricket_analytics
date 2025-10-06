"""
RAG (Retrieval-Augmented Generation) Module for Cricket Analytics
"""

import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain.schema import Document, BaseMessage
from langchain_community.llms import OpenAI
try:
    from langchain_community.llms import Groq  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    Groq = None  # type: ignore[assignment]

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers for the cricket RAG system."""

    OPENAI = "openai"
    GROQ = "groq"
    GEMINI = "gemini"


DEFAULT_LLM_MODELS: Dict[str, str] = {
    LLMProvider.OPENAI.value: "gpt-3.5-turbo",
    LLMProvider.GROQ.value: "mixtral-8x7b-32768",
    LLMProvider.GEMINI.value: "gemini-pro",
}


ENV_API_KEY_MAP: Dict[str, List[str]] = {
    LLMProvider.OPENAI.value: ["OPENAI_API_KEY"],
    LLMProvider.GROQ.value: ["GROQ_API_KEY"],
    LLMProvider.GEMINI.value: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}

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

    def __init__(
        self,
        llm_provider: str = LLMProvider.OPENAI.value,
        *,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
    ):

        self.llm_provider = self._normalize_provider(llm_provider)
        self.temperature = temperature
        self.model_name = model_name or DEFAULT_LLM_MODELS[self.llm_provider]
        self.api_keys: Dict[str, Optional[str]] = {
            LLMProvider.OPENAI.value: openai_api_key,
            LLMProvider.GROQ.value: groq_api_key,
            LLMProvider.GEMINI.value: gemini_api_key,
        }

        self.llm = self._initialize_llm()

        self.embedding_manager = CricketEmbeddingManager()
        self.vectorstore = self.embedding_manager.load_existing_vectorstore()

        if self.vectorstore:
            self.retriever = CricketRetriever(self.vectorstore)
        else:
            self.retriever = None
            logger.warning("No vector store available. Call initialize_vectorstore() first.")

        self.qa_chain = None
        self._setup_qa_chain()

    def _normalize_provider(self, provider: Optional[str]) -> str:
        normalized = (provider or LLMProvider.OPENAI.value).lower()
        if normalized not in DEFAULT_LLM_MODELS:
            logger.warning(
                "Unsupported LLM provider '%s'. Falling back to '%s'.",
                normalized,
                LLMProvider.OPENAI.value,
            )
            return LLMProvider.OPENAI.value
        return normalized

    def _get_api_key(self, provider: str) -> Optional[str]:
        explicit = self.api_keys.get(provider)
        if explicit:
            return explicit
        for env_key in ENV_API_KEY_MAP.get(provider, []):
            env_value = os.getenv(env_key)
            if env_value:
                return env_value
        return None

    def _initialize_llm(self) -> Optional[Any]:
        api_key = self._get_api_key(self.llm_provider)
        if not api_key:
            logger.warning(
                "%s API key not provided. LLM-powered features will be disabled.",
                self.llm_provider.upper(),
            )
            return None

        try:
            if self.llm_provider == LLMProvider.OPENAI.value:
                llm_instance = OpenAI(
                    openai_api_key=api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                )
            elif self.llm_provider == LLMProvider.GROQ.value:
                if Groq is None:
                    logger.error(
                        "Groq provider requested but 'groq' / langchain community integration is missing. "
                        "Install the groq extra dependencies to enable it."
                    )
                    return None
                llm_instance = Groq(
                    groq_api_key=api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                )
            elif self.llm_provider == LLMProvider.GEMINI.value:
                if ChatGoogleGenerativeAI is None:
                    logger.error(
                        "Gemini provider requested but 'langchain-google-genai' is not installed."
                    )
                    return None
                llm_instance = ChatGoogleGenerativeAI(
                    api_key=api_key,
                    model=self.model_name,
                    temperature=self.temperature,
                    convert_system_message_to_human=True,
                )
            else:
                logger.error("Unsupported LLM provider '%s'.", self.llm_provider)
                return None

            logger.info(
                "Initialized %s provider with model '%s'.",
                self.llm_provider,
                self.model_name,
            )
            return llm_instance

        except Exception as exc:  # pragma: no cover - network/provider errors
            logger.error(f"Failed to initialize {self.llm_provider} LLM: {exc}")
            return None

    def _invoke_llm(self, prompt: str) -> str:
        if not self.llm:
            raise ValueError("LLM is not configured. Please provide valid API credentials.")

        try:
            if hasattr(self.llm, "predict"):
                return self.llm.predict(prompt)  # type: ignore[no-any-return]

            response: Union[str, BaseMessage, Any] = (
                self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)  # type: ignore[call-arg]
            )

            if isinstance(response, BaseMessage):
                return str(response.content)
            if isinstance(response, str):
                return response
            if isinstance(response, dict) and "content" in response:
                return str(response["content"])  # Fallback for dict responses
            return str(response)

        except Exception as exc:  # pragma: no cover - network/provider errors
            logger.error(f"Failed to invoke {self.llm_provider} LLM: {exc}")
            raise
    
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
                    "answer": (
                        "LLM is not configured. Please provide valid API credentials for the selected "
                        f"provider ('{self.llm_provider}') to enable generated insights."
                    ),
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
            
            answer = self._invoke_llm(prompt)
            
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
def setup_cricket_rag(
    data_directory: str = "data/processed",
    *,
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> Optional[CricketRAG]:
    """Setup and initialize cricket RAG system with configurable LLM provider."""

    try:
        provider = llm_provider or os.getenv("LLM_PROVIDER") or LLMProvider.OPENAI.value

        resolved_temperature: float
        if temperature is not None:
            resolved_temperature = temperature
        else:
            temp_env = os.getenv("LLM_TEMPERATURE")
            if temp_env:
                try:
                    resolved_temperature = float(temp_env)
                except ValueError:
                    logger.warning(
                        "Invalid LLM_TEMPERATURE value '%s'. Defaulting to 0.3.",
                        temp_env,
                    )
                    resolved_temperature = 0.3
            else:
                resolved_temperature = 0.3

        rag = CricketRAG(
            llm_provider=provider,
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
            gemini_api_key=
                gemini_api_key
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY"),
            model_name=model_name or os.getenv("LLM_MODEL_NAME"),
            temperature=resolved_temperature,
        )
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