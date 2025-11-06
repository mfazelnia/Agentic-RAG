"""
Main script for the Agentic RAG System.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.document_loader import load_documents, chunk_documents
from src.vector_store import VectorStore
from src.simple_rag import SimpleRAG
from src.agentic_rag import AgenticRAG


def main():
    """Main entry point for the RAG system."""
    
    print("=== Agentic RAG System ===\n")
    
    # Load and chunk documents
    data_dir = "data/sample_docs"
    print(f"Loading documents from {data_dir}...")
    
    try:
        documents = load_documents(data_dir)
        if not documents:
            print("No documents found. Please add documents to data/sample_docs/")
            return
        
        print(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        chunked_docs = chunk_documents(documents, chunk_size=500, overlap=50)
        print(f"Created {len(chunked_docs)} document chunks\n")
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    # Build vector store
    try:
        vector_store = VectorStore()
        vector_store.add_documents(chunked_docs)
        print()
    except Exception as e:
        print(f"Error building vector store: {e}")
        return
    
    # Initialize RAG system (use agentic by default)
    try:
        rag = AgenticRAG(vector_store, max_iterations=3)
        print("Using agentic RAG mode (with planning, iteration, and reflection)")
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        print("Make sure you have a .env file with OPENAI_API_KEY set")
        return
    
    # Interactive Q&A loop
    print("\nReady! Ask questions about the documents.")
    print("Type 'exit' to quit, 'verbose' to toggle detailed output.\n")
    
    verbose = False
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'verbose':
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}\n")
                continue
            
            if not query:
                continue
            
            if verbose:
                print("\nProcessing query...")
            else:
                print("\nSearching...")
            
            result = rag.query(query, k=5, verbose=verbose)
            
            print(f"\nAnswer: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources: {', '.join(result['sources'])}")
            
            if verbose and result.get('iterations'):
                print(f"\nIterations used: {len(result['iterations'])}")
                print(f"Total documents retrieved: {result.get('total_docs_used', 0)}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

