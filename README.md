# Agentic RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions based on your documents using vector search and OpenAI's GPT models.

## Features

- 📄 Load and process text files and PDFs
- 🔍 Semantic search using FAISS and sentence transformers
- 🤖 Generate answers using OpenAI's GPT-4o-mini
- 💬 Interactive CLI for Q&A
- 📊 Source attribution for answers

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Agentic_RAG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
cp env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

1. Add your documents to `data/sample_docs/`

2. Run the application:
```bash
python main.py
```

3. Ask questions in the interactive prompt:
```
Query: What is machine learning?
```

4. Type 'exit' to quit

## Project Structure

```
agentic-rag-system/
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading and chunking
│   ├── vector_store.py        # FAISS vector store
│   └── simple_rag.py          # RAG orchestration
├── data/
│   └── sample_docs/           # Your documents go here
├── requirements.txt
├── env.example                 # Environment template
├── .env                        # API keys (create from env.example)
├── README.md
└── main.py                     # Main entry point
```

## How It Works

1. **Document Loading**: Documents are loaded from `data/sample_docs/` and split into chunks
2. **Embedding**: Each chunk is converted to a vector using sentence transformers
3. **Indexing**: Vectors are stored in FAISS for fast similarity search
4. **Querying**: When you ask a question, the system retrieves relevant chunks
5. **Generation**: OpenAI generates an answer based on the retrieved context
6. **Sources**: The system shows which documents were used

## Requirements

- Python 3.8+
- OpenAI API key
- Documents in `data/sample_docs/`

## License

MIT

