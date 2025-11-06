# Agentic RAG System

A RAG implementation for Q/A on your documents. It uses vector search to find relevant content and OpenAI's GPT models to generate answers.

## Features

- Load and process text files and PDFs
- Semantic search using FAISS and sentence transformers
- Generate answers using OpenAI's GPT-4o-mini
- Interactive CLI for asking questions
- Shows which documents were used to answer each question

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mfazelnia/Agentic-RAG.git
cd Agentic-RAG
```

2. Create a virtual environment:
```bash
python -m venv your_venv_name
source your_venv_name/bin/activate 
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
cp env.example .env
# Then edit .env and add your OPENAI_API_KEY
```

## Usage

Put your documents in the `data/sample_docs/` folder, then run:

```bash
python main.py
```

You'll get an interactive prompt where you can ask questions. Type 'exit' when you're done.

## Project Structure

```
agentic-rag-system/
├── src/
│   ├── __init__.py
│   ├── document_loader.py    
│   ├── vector_store.py       
│   └── simple_rag.py         
├── data/
│   └── sample_docs/          
├── requirements.txt
├── env.example               
├── .env                        
├── README.md
└── main.py                     
```

## How It Works

The system loads documents from data/sample_docs/ and splits them into chunks. Each chunk gets converted to a vector using sentence transformers, and these vectors are stored in FAISS for fast similarity search. When a query comes in, it finds the relevant chunks and uses an LLM to generate responses. It also shows which documents were used.

## Requirements

Python 3.8 or higher and an OpenAI API key. Put your documents in `data/sample_docs/`.

## License

MIT

