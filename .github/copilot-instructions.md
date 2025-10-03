# Cricket Analytics RAG Project - Copilot Instructions

This project implements an Agentic AI system for cricket match analytics using RAG + LangChain for multimodal data analysis.

## Project Overview
- **Goal**: Enhance cricket strategy insights by combining structured stats with unstructured commentary
- **Tech Stack**: Python, LangChain, LangGraph, Vector Database, RAG
- **Data Sources**: Cricinfo commentary, Kaggle ball-by-ball datasets, match reports

## Key Components
1. **Data Pipeline**: Ingest and process cricket data from multiple sources
2. **RAG System**: Retrieve and generate insights from multimodal cricket data
3. **LangGraph**: Multi-hop reasoning for complex tactical queries
4. **API Layer**: Serve cricket insights through REST endpoints
5. **Evaluation**: Accuracy testing against ground truth data

## Development Guidelines
- Focus on cricket domain expertise in prompts and responses
- Prioritize data quality and preprocessing for cricket-specific metrics
- Implement robust error handling for varied data formats
- Use cricket terminology and context appropriately
- Ensure scalability for large cricket datasets

## Example Queries to Support
- "How does Virat Kohli perform against left-arm pacers in the death overs?"
- "What bowling changes worked best in the last 5 matches at Eden Gardens?"
- Player performance analysis across different conditions and opponents

✅ **Step 1: Clarify Project Requirements** - COMPLETED
✅ **Step 2: Scaffold the Project** - COMPLETED
✅ **Step 3: Install Dependencies** - COMPLETED
   - Created Jupyter notebook with installation steps
   - Added essential package installation code
   - Fixed import errors for plotly, dotenv, and LangChain
   - Resolved LangChain/LangSmith version conflicts and installed full requirements set
✅ **Step 4: Configure Environment** - COMPLETED
   - Created configuration files (rag_config.yaml)
   - Set up environment variables
   - Created sample cricket data for testing
✅ **Step 5: Compile & Test** - COMPLETED
   - Resolved LangChain dependency alignment and community module imports
   - Migrated FastAPI lifecycle to lifespan handler to clear deprecation warnings
   - Added baseline pytest coverage for RAG examples and validated suite