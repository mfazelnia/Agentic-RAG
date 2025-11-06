"""
Agentic RAG module with multi-step reasoning, iterative refinement, and self-reflection.
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

from vector_store import VectorStore

# Load environment variables
load_dotenv()


class AgenticRAG:
    """Agentic RAG system with planning, iteration, and reflection capabilities."""
    
    def __init__(self, vector_store: VectorStore, max_iterations: int = 3):
        """
        Initialize agentic RAG system.
        
        Args:
            vector_store: Initialized VectorStore instance
            max_iterations: Maximum number of search-refine iterations
        """
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = 'gpt-4o-mini'
    
    def _plan_query(self, query: str) -> Dict:
        """
        Plan the query by breaking it down into sub-queries if needed.
        
        Args:
            query: Original user query
            
        Returns:
            Dictionary with plan and sub-queries
        """
        planning_prompt = f"""Analyze this query and determine if it needs to be broken down into sub-queries for better retrieval.

Query: {query}

Respond in JSON format:
{{
    "needs_decomposition": true/false,
    "reasoning": "brief explanation",
    "sub_queries": ["sub-query 1", "sub-query 2", ...] or []
}}

If the query is simple and straightforward, set needs_decomposition to false and sub_queries to an empty array.
If the query is complex (e.g., comparing multiple concepts, multi-part questions), break it down."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query planning assistant. Always respond with valid JSON only."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
        except Exception as e:
            # Fallback: treat as simple query
            return {
                "needs_decomposition": False,
                "reasoning": f"Planning failed: {str(e)}",
                "sub_queries": []
            }
    
    def _search_and_retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        return self.vector_store.search(query, k=k)
    
    def _check_completeness(self, query: str, answer: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Check if the answer is complete and if more information is needed.
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_docs: Documents used to generate answer
            
        Returns:
            Dictionary with completeness assessment
        """
        context_preview = "\n".join([doc['text'][:200] for doc in retrieved_docs[:3]])
        
        reflection_prompt = f"""Evaluate if this answer fully addresses the query. Consider:
1. Does it answer all parts of the question?
2. Are there gaps or missing information?
3. Would additional searches help?

Query: {query}

Answer: {answer}

Retrieved context preview:
{context_preview}

Respond in JSON format:
{{
    "is_complete": true/false,
    "confidence": "high/medium/low",
    "missing_aspects": ["aspect 1", "aspect 2", ...] or [],
    "needs_refinement": true/false,
    "refinement_query": "follow-up query if needs_refinement is true" or ""
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quality assessment assistant. Always respond with valid JSON only."},
                    {"role": "user", "content": reflection_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            assessment = json.loads(response.choices[0].message.content)
            return assessment
        except Exception as e:
            # Fallback: assume complete
            return {
                "is_complete": True,
                "confidence": "medium",
                "missing_aspects": [],
                "needs_refinement": False,
                "refinement_query": ""
            }
    
    def _generate_answer(self, query: str, contexts: List[str], sources: List[str], iteration: int = 0) -> str:
        """
        Generate an answer from retrieved contexts.
        
        Args:
            query: User query
            contexts: List of context strings from retrieved documents
            sources: List of source document names
            iteration: Current iteration number
            
        Returns:
            Generated answer
        """
        context_text = "\n\n".join([
            f"Context {i+1} (from {sources[i]}):\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
        
        if iteration > 0:
            prompt = f"""You are refining your answer based on additional context. Use the following documents to provide a comprehensive answer.

Previous iteration: {iteration}
Context from documents:
{context_text}

Question: {query}

Provide a complete answer that synthesizes all the information:"""
        else:
            prompt = f"""Use the following documents to answer the question. If the answer is not in the documents, say so.

Context from documents:
{context_text}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents. Be thorough and accurate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, query: str, k: int = 5, verbose: bool = False) -> Dict:
        """
        Query the agentic RAG system with planning, iteration, and reflection.
        
        Args:
            query: User query
            k: Number of documents to retrieve per search
            verbose: Whether to print intermediate steps
            
        Returns:
            Dictionary with 'answer', 'sources', 'iterations', and 'reasoning'
        """
        all_retrieved_docs = []
        all_sources = set()
        iterations_used = []
        
        # Step 1: Plan the query
        if verbose:
            print("Planning query...")
        plan = self._plan_query(query)
        
        if verbose and plan.get('needs_decomposition'):
            print(f"Query decomposed into {len(plan.get('sub_queries', []))} sub-queries")
        
        # Step 2: Execute searches
        queries_to_search = plan.get('sub_queries', []) if plan.get('needs_decomposition') else [query]
        
        for search_query in queries_to_search:
            if verbose:
                print(f"Searching: {search_query}")
            docs = self._search_and_retrieve(search_query, k=k)
            all_retrieved_docs.extend(docs)
            all_sources.update([doc['source'] for doc in docs])
        
        # Remove duplicates while preserving order
        seen_texts = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            if doc['text'] not in seen_texts:
                seen_texts.add(doc['text'])
                unique_docs.append(doc)
        
        all_retrieved_docs = unique_docs[:k*2]  # Limit total docs
        
        # Step 3: Generate initial answer
        contexts = [doc['text'] for doc in all_retrieved_docs]
        sources_list = [doc['source'] for doc in all_retrieved_docs]
        
        answer = self._generate_answer(query, contexts, sources_list, iteration=0)
        iterations_used.append({
            'iteration': 0,
            'docs_retrieved': len(all_retrieved_docs),
            'answer': answer
        })
        
        # Step 4: Iterative refinement
        for iteration in range(1, self.max_iterations):
            if verbose:
                print(f"Reflecting on answer (iteration {iteration})...")
            
            assessment = self._check_completeness(query, answer, all_retrieved_docs)
            
            if not assessment.get('needs_refinement', False):
                if verbose:
                    print(f"Answer is complete (confidence: {assessment.get('confidence', 'unknown')})")
                break
            
            refinement_query = assessment.get('refinement_query', '')
            if not refinement_query:
                break
            
            if verbose:
                print(f"Refining search: {refinement_query}")
            
            # Search for additional information
            additional_docs = self._search_and_retrieve(refinement_query, k=k//2)
            
            # Add new docs if they're different
            for doc in additional_docs:
                if doc['text'] not in seen_texts:
                    seen_texts.add(doc['text'])
                    all_retrieved_docs.append(doc)
                    all_sources.add(doc['source'])
            
            # Regenerate answer with all context
            contexts = [doc['text'] for doc in all_retrieved_docs[:k*2]]
            sources_list = [doc['source'] for doc in all_retrieved_docs[:k*2]]
            
            answer = self._generate_answer(query, contexts, sources_list, iteration=iteration)
            iterations_used.append({
                'iteration': iteration,
                'docs_retrieved': len(all_retrieved_docs),
                'refinement_query': refinement_query,
                'answer': answer
            })
        
        return {
            'answer': answer,
            'sources': list(all_sources),
            'retrieved_docs': all_retrieved_docs,
            'iterations': iterations_used,
            'plan': plan,
            'total_docs_used': len(all_retrieved_docs)
        }

