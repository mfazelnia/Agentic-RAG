"""
Simple RAG module that combines retrieval with generation.
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

from vector_store import VectorStore

# Load environment variables
load_dotenv()


class SimpleRAG:
    """Simple RAG system combining retrieval and generation."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG system.
        
        Args:
            vector_store: Initialized VectorStore instance
        """
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = 'gpt-4o-mini'
    
    def query(self, query: str, k: int = 3) -> Dict:
        """
        Query the RAG system.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, k=k)
        
        if not retrieved_docs:
            return {
                'answer': "No relevant documents found.",
                'sources': []
            }
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} (from {doc['source']}):\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Build prompt
        prompt = f"""Use the following documents to answer the question. If the answer is not in the documents, say so.

Context from documents:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Extract sources
            sources = list(set([doc['source'] for doc in retrieved_docs]))
            
            return {
                'answer': answer,
                'sources': sources,
                'retrieved_docs': retrieved_docs
            }
        
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': []
            }

