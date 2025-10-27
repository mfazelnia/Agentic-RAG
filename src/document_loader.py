"""
Document loader module for loading and chunking documents.
"""

import os
import glob
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader


def load_documents(directory: str) -> List[Dict]:
    """
    Load text files and PDFs from a directory.
    
    Args:
        directory: Path to the directory containing documents
        
    Returns:
        List of dictionaries with 'text' and 'source' keys
    """
    documents = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    # Load text files
    for txt_file in glob.glob(os.path.join(directory, "*.txt")):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'text': content,
                    'source': os.path.basename(txt_file)
                })
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    
    # Load PDF files
    for pdf_file in glob.glob(os.path.join(directory, "*.pdf")):
        try:
            reader = PdfReader(pdf_file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            
            documents.append({
                'text': content,
                'source': os.path.basename(pdf_file)
            })
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    return documents


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in tokens (words)
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Split documents into chunks with metadata.
    
    Args:
        documents: List of document dictionaries with 'text' and 'source'
        chunk_size: Size of each chunk in tokens
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunked_docs = []
    
    for doc in documents:
        chunks = split_text(doc['text'], chunk_size, overlap)
        
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                'text': chunk,
                'source': doc['source'],
                'chunk_index': idx
            })
    
    return chunked_docs

