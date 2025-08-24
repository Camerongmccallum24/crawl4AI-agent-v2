"""Utility functions for text processing and Supabase operations."""

import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize Supabase client
def get_supabase_client(use_service_role: bool = False) -> Client:
    """Get a Supabase client using environment variables."""
    # Force reload environment variables to avoid caching issues
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    url = os.getenv("SUPABASE_URL")
    if use_service_role:
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    else:
        key = os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and appropriate key must be set")
    return create_client(url, key)

# Initialize embedding model
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get the embedding model."""
    return SentenceTransformer(model_name)

def add_documents_to_supabase(
    table_name: str,
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 100,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> None:
    """Add documents to Supabase table in batches."""
    client = get_supabase_client(use_service_role=True)  # Use service role to bypass RLS
    model = get_embedding_model(embedding_model_name)
    
    if metadatas is None:
        metadatas = [{}] * len(documents)
    
    # Process in batches
    for i in range(0, len(documents), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        # Generate embeddings
        embeddings = model.encode(batch_docs).tolist()
        
        # Prepare data for insertion
        data = []
        for j, (doc_id, doc, metadata, embedding) in enumerate(
            zip(batch_ids, batch_docs, batch_metadatas, embeddings)
        ):
            data.append({
                "id": doc_id,
                "content": doc,
                "metadata": metadata,
                "embedding": embedding
            })
        
        # Insert batch
        client.table(table_name).insert(data).execute()

def query_supabase(
    table_name: str,
    query_text: str,
    n_results: int = 5,
    where_filter: Optional[Dict[str, Any]] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """Query Supabase table for similar documents."""
    client = get_supabase_client()
    model = get_embedding_model(embedding_model_name)
    
    # Generate query embedding
    query_embedding = model.encode([query_text]).tolist()[0]
    
    # Use the custom function for similarity search
    result = client.rpc('match_documents', {
        'query_embedding': query_embedding,
        'match_threshold': 0.0,
        'match_count': n_results
    }).execute()
    
    return {
        "documents": [row["content"] for row in result.data],
        "metadatas": [row["metadata"] for row in result.data],
        "ids": [row["id"] for row in result.data],
        "distances": [1 - row["similarity"] for row in result.data]  # Convert similarity to distance
    }

def format_results_as_context(query_results: Dict[str, Any]) -> str:
    """Format query results as a context string for the agent.
    
    Args:
        query_results: Results from a Supabase query
        
    Returns:
        Formatted context string
    """
    context = "CONTEXT INFORMATION:\n\n"
    
    for i, (doc, metadata, distance) in enumerate(zip(
        query_results["documents"],
        query_results["metadatas"],
        query_results["distances"]
    )):
        # Add document information
        context += f"Document {i+1} (Relevance: {1 - distance:.2f}):\n"
        
        # Add metadata if available
        if metadata:
            for key, value in metadata.items():
                context += f"{key}: {value}\n"
        
        # Add document content
        context += f"Content: {doc}\n\n"
    
    return context
