# Cricket Analytics RAG Project

An Agentic AI system for cricket match analytics using RAG + LangChain for multimodal data analysis.

## 🏏 Project Overview

This project enhances cricket strategy insights by combining structured statistics with unstructured commentary, match reports, and analysis using advanced RAG (Retrieval-Augmented Generation) techniques.

### Key Features
- **Multimodal Data Integration**: Combines structured stats with unstructured commentary
- **Advanced RAG System**: LangChain-powered retrieval and generation
- **Multi-hop Reasoning**: LangGraph for complex tactical queries
- **Cricket Domain Expertise**: Specialized for cricket analytics and strategy

### Example Queries
- "How does Virat Kohli perform against left-arm pacers in the death overs?"
- "What bowling changes worked best in the last 5 matches at Eden Gardens?"
- "Analyze team performance in chase scenarios under 15 overs"

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key (or other LLM provider)
- Vector database (ChromaDB/FAISS)

### Installation

1. **Clone and Setup**
```bash
git clone <repository-url>
cd cricket-analytics-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```
5. **Initialize Vector Store (Demo Data)**
```powershell
# Windows PowerShell
$env:RAG_DATA_DIR = "data/sample"
python .\scripts\init_vectorstore.py
```

6. **Run the API**
```powershell
# Choose provider and set key (example: Groq)
$env:LLM_PROVIDER = "groq"; $env:GROQ_API_KEY = "<your-groq-key>"

# Start
.\.venv\Scripts\uvicorn.exe src.api:app --reload
```

Visit http://127.0.0.1:8000/docs to try endpoints.

### Choose Your LLM Provider

Set the `LLM_PROVIDER` environment variable to `openai`, `groq`, or `gemini`, then supply the matching API key. Examples below use PowerShell (`$env:`) and Bash (`export`).

```powershell
# PowerShell (Windows)
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-groq-key"
```

```bash
# Bash / zsh (macOS, Linux)
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your-gemini-key"
```

If you stick with OpenAI, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` as before. You can also override the default model via `LLM_MODEL_NAME` and temperature with `LLM_TEMPERATURE` when needed.

3. **Data Setup**
```bash
# Download sample cricket datasets
python src/data/download_data.py
```

4. **Initialize Vector Database**
```bash
python src/rag/initialize_embeddings.py
```

## 📁 Project Structure

```
cricket-analytics-rag/
├── src/
│   ├── data/                 # Data ingestion and processing
│   │   ├── collectors/       # Data collection from various sources
│   │   ├── processors/       # Data cleaning and transformation
│   │   └── loaders/         # Data loading utilities
│   ├── rag/                 # RAG implementation
│   │   ├── retrievers/      # Document retrieval systems
│   │   ├── generators/      # Response generation
│   │   └── embeddings/      # Embedding management
│   ├── langgraph/           # Multi-hop reasoning
│   │   ├── nodes/           # Graph nodes for different reasoning steps
│   │   ├── edges/           # Graph edges and flow control
│   │   └── workflows/       # Complete reasoning workflows
│   ├── api/                 # FastAPI endpoints
│   └── evaluation/          # Testing and evaluation
├── data/
│   ├── raw/                 # Raw cricket data
│   └── processed/           # Processed and cleaned data
├── notebooks/               # Jupyter notebooks for experimentation
├── config/                  # Configuration files
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## 🏏 Data Sources

### Structured Data
- **Ball-by-ball data**: Kaggle cricket datasets
- **Match statistics**: Player performance metrics
- **Ground information**: Venue-specific data

### Unstructured Data
- **Commentary**: Cricinfo match commentaries
- **Match reports**: Post-match analysis and reports
- **Player interviews**: Quotes and insights

## 🔧 Usage

### 1. Basic Query
```python
from src.rag import CricketRAG

rag = CricketRAG()
response = rag.query("How does Virat Kohli perform in pressure situations?")
print(response)
```

### 2. Multi-hop Reasoning
```python
from src.langgraph import CricketAnalyzer

analyzer = CricketAnalyzer()
insights = analyzer.analyze_complex_query(
    "What tactical changes should be made for the next match based on recent performance?"
)
```

### 3. API Usage
```bash
# Start the API server
uvicorn src.api.main:app --reload

# Query via REST API
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Analyze batting performance in powerplay overs"}'
```

## 📊 Development Timeline

### Week 5-7: Data Foundation
- [x] Dataset collection and exploration
- [x] Data preprocessing pipeline
- [x] Initial data quality assessment

### Week 9: Baseline RAG
- [ ] Basic RAG implementation
- [ ] Vector database setup
- [ ] Simple query-response system

### Week 11: Advanced Reasoning
- [ ] LangGraph integration
- [ ] Multi-hop reasoning capabilities
- [ ] Complex tactical analysis

### Evaluation
- [ ] Ground truth validation
- [ ] F1-score metrics
- [ ] Expert cricket knowledge testing

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_rag.py
pytest tests/test_langgraph.py

# Run with coverage
pytest --cov=src tests/
```

## 📈 Evaluation Metrics

- **Accuracy**: Against ground truth cricket data
- **F1-Score**: For information retrieval quality
- **Expert Validation**: Cricket domain expert review
- **Response Relevance**: Query-answer alignment

## 🛠️ Configuration

Key configuration files in `config/`:
- `rag_config.yaml`: RAG system parameters
- `model_config.yaml`: LLM and embedding model settings
- `data_config.yaml`: Data source configurations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏏 Cricket Domain Notes

This system is specifically designed for cricket analytics and uses cricket-specific terminology, metrics, and domain knowledge. The AI responses are optimized for cricket strategy and tactical insights.

---

**Happy Cricket Analytics! 🏏**