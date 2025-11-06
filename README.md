# Agentic RAG System

A RAG implementation for Q/A on your documents. It uses vector search to find relevant content and OpenAI's GPT models to generate answers.

## Features

- Load and process text files and PDFs
- Semantic search using FAISS and sentence transformers
- Generate answers using OpenAI's GPT-4o-mini
- Interactive CLI for asking questions
- Shows which documents were used to answer each question
- Query planning: breaks down complex queries into sub-queries
- Iterative refinement: multiple search cycles to improve answers
- Self-reflection: evaluates answer completeness and quality

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

You'll get an interactive prompt where you can ask questions. Type 'exit' when you're done. Type 'verbose' to toggle detailed output showing the planning, reflection, and iteration steps.

## Project Structure

```
agentic-rag-system/
├── src/
│   ├── __init__.py
│   ├── document_loader.py    
│   ├── vector_store.py       
│   ├── simple_rag.py         
│   └── agentic_rag.py        
├── data/
│   └── sample_docs/          
├── requirements.txt
├── env.example               
├── .env                        
├── README.md
└── main.py                     
```

## How It Works

The system loads documents from data/sample_docs/ and splits them into chunks. Each chunk gets converted to a vector using sentence transformers, and these vectors are stored in FAISS for fast similarity search. 

When a query comes in, the agentic system follows these steps:

1. **Query Planning**: Analyzes the query and breaks it down into sub-queries if needed
2. **Initial Retrieval**: Searches for relevant chunks using vector similarity
3. **Answer Generation**: Uses an LLM to generate an answer from retrieved context
4. **Self-Reflection**: Evaluates if the answer is complete and identifies gaps
5. **Iterative Refinement**: Performs additional searches if needed and refines the answer
6. **Final Answer**: Returns the refined answer with source attribution

The system can perform up to 3 iterations to ensure comprehensive answers.

## Agentic Capabilities

### Query Planning

The system analyzes incoming queries to determine if they need to be decomposed into multiple sub-queries. This is especially useful for complex questions that require information from different aspects or multiple documents. For example, a query like "Compare machine learning and neural networks" would be broken down into separate searches for each concept.

### Iterative Refinement

Instead of a single search-and-answer cycle, the system can perform multiple iterations. After generating an initial answer, it checks if more information is needed. If gaps are identified, it performs follow-up searches with refined queries and synthesizes the new information into a more complete answer.

### Self-Reflection

After generating an answer, the system evaluates its own output to assess:
- Whether all parts of the question are answered
- The confidence level of the answer
- Missing aspects that might need additional searches
- Whether refinement is necessary

This self-assessment capability allows the system to improve its responses autonomously.

## Requirements

Python 3.8 or higher and an OpenAI API key. Put your documents in `data/sample_docs/`.

## License

MIT

