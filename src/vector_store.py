"""
Vector store module for managing document embeddings and similarity search.
"""

from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Simple vector store using FAISS and sentence-transformers."""
    
    def __init__(self):
        """Initialize the vector store with embedding model."""
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        self.index = None
        self.documents = []
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and metadata
        """
        if not documents:
            raise ValueError("No documents to add")
        
        # Extract text from documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Initialize or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = documents.copy()
        else:
            self.documents.extend(documents)
        
        self.index.add(embeddings)
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document dictionaries with text and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                **self.documents[idx],
                'distance': float(distance)
            })
        
        return results

