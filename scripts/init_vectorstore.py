import os
from pathlib import Path
from dotenv import load_dotenv
from src.rag import setup_cricket_rag


def main():
    load_dotenv()
    data_dir = os.getenv("RAG_DATA_DIR", "data/sample")
    rag = setup_cricket_rag(data_directory=data_dir)
    if rag and rag.vectorstore:
        print(f"Vectorstore initialized from: {data_dir}")
    else:
        raise SystemExit("Failed to initialize vectorstore. Check data dir and dependencies.")


if __name__ == "__main__":
    main()
