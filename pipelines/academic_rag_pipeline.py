"""
title: Academic RAG Pipeline
author: Continuum Labs
version: 1.0
description: RAG pipeline for academic paper search and analysis
requirements: pymilvus,requests,pydantic
"""

from __future__ import annotations
from typing import (
    List, 
    Union, 
    Generator, 
    Iterator, 
    Optional, 
    Dict,
    Tuple,
    Any
)
from pydantic import BaseModel
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pymilvus import connections, Collection, utility
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        # Milvus Configuration
        MILVUS_HOST: str = os.getenv("MILVUS_HOST", "10.106.175.99")
        MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
        MILVUS_USER: str = os.getenv("MILVUS_USER", "thannon")
        MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "chaeBio7!!!")
        MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "arxiv_documents")
        
        # Embedding Configuration
        EMBEDDING_ENDPOINT: str = os.getenv("EMBEDDING_ENDPOINT", "http://192.168.13.50:30000/v1/embeddings")
        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-mistral-7b-v2")
        
        # Search Configuration
        TOP_K: int = 5
        SCORE_THRESHOLD: float = 2.0  # Added type clarity with .0
        
        # System Prompt
        SYSTEM_PROMPT: str = """You are a highly knowledgeable research analyst with expertise in scientific papers. 
        When answering questions:
        1. Start with a brief 1-2 sentence summary of your findings
        2. Cite specific papers with their years when making key points
        3. Compare findings across papers when possible
        4. Explain any technical terms in parentheses
        5. Structure your response as:
           • Summary
           • Key Findings (with citations)
           • Technical Details
           • References

        Keep responses clear and focused on the most relevant information."""

        class Config:
            arbitrary_types_allowed = True

    def __init__(self) -> None:
        """Initialize the Pipeline with necessary connections and configurations."""
        try:
            self.valves = self.Valves()
            self.session = self._initialize_session()
            self._initialize_milvus()
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise

    def _initialize_session(self) -> requests.Session:
        """Set up session with retry strategy for embedding requests."""
        try:
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logger.info("HTTP session initialized with retry strategy")
            return session
        except Exception as e:
            logger.error(f"Failed to initialize HTTP session: {str(e)}")
            raise

    def _initialize_milvus(self) -> None:
        """Connect to Milvus and initialize the collection."""
        try:
            connections.connect(
                alias="default",
                host=self.valves.MILVUS_HOST,
                port=self.valves.MILVUS_PORT,
                user=self.valves.MILVUS_USER,
                password=self.valves.MILVUS_PASSWORD
            )
            
            collection_name = self.valves.MILVUS_COLLECTION
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist in Milvus.")
            
            self._collection = Collection(collection_name)
            self._collection.load()
            logger.info(f"Connected to Milvus and loaded collection '{collection_name}'")
            
            if not self._collection.is_empty:
                entity_count = self._collection.num_entities
                logger.info(f"Collection loaded with {entity_count} entities")
            else:
                logger.warning("Collection is empty")
                
        except Exception as e:
            logger.error(f"Failed to initialize Milvus connection: {str(e)}")
            raise

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate an embedding for the given text."""
        try:
            payload = {
                "input": [text],
                "model": self.valves.EMBEDDING_MODEL,
                "input_type": "query",
                "truncate": "END"
            }
            
            response = self.session.post(
                self.valves.EMBEDDING_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            embedding = data.get("data", [])[0].get("embedding", [])
            
            if len(embedding) != 4096:
                logger.error(f"Invalid embedding dimension: {len(embedding)}")
                return None
            
            logger.info(f"Generated embedding of dimension {len(embedding)}")
            return embedding
            
        except requests.RequestException as e:
            logger.error(f"Error fetching embedding: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in embedding generation: {str(e)}")
            return None

    def search_documents(self, embedding: List[float]) -> List[Dict[str, Any]]:
        """Search Milvus for similar documents based on embedding."""
        try:
            logger.info(f"Searching collection with {len(embedding)}-dimensional vector")
            
            results = self._collection.search(
                data=[embedding],
                anns_field="embedding",
                param={
                    "metric_type": "L2",
                    "params": {
                        "search_width": 128,
                        "ef_search": 100,
                        "nprobe": 16
                    }
                },
                limit=self.valves.TOP_K,
                output_fields=[
                    "doc_id", "source_file", "text", "summary", 
                    "abstract", "key_points", "technical_terms", 
                    "relationships"
                ]
            )
            
            logger.info(f"Search returned {len(results)} result sets")
            documents: List[Dict[str, Any]] = []
            
            for hits in results:
                for hit in hits:
                    score = float(hit.distance)
                    if score <= self.valves.SCORE_THRESHOLD:
                        source_file = getattr(hit.entity, "source_file", "Unknown")
                        year = source_file.split('-')[0] if source_file.startswith('20') else 'Unknown'
                        
                        document = {
                            "doc_id": getattr(hit.entity, "doc_id", -1),
                            "source_file": source_file,
                            "year": year,
                            "text": getattr(hit.entity, "text", ""),
                            "summary": getattr(hit.entity, "summary", ""),
                            "abstract": getattr(hit.entity, "abstract", ""),
                            "key_points": getattr(hit.entity, "key_points", ""),
                            "technical_terms": getattr(hit.entity, "technical_terms", ""),
                            "relationships": getattr(hit.entity, "relationships", ""),
                            "score": score
                        }
                        documents.append(document)
                        logger.info(f"Added document '{source_file}' ({year}) with score {score}")
            
            if not documents:
                logger.warning("No documents passed the score threshold.")
            return documents
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def format_context(self, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Format documents optimally for LLM and user."""
        llm_context: List[str] = []
        user_metadata: List[Dict[str, Any]] = []
        
        for i, doc in enumerate(documents, 1):
            llm_section = (
                f"Document {i} ({doc['year']}): {doc['source_file']}\n\n"
                f"Abstract:\n{doc['abstract'][:500]}...\n\n"
                f"Key Points:\n{self.format_bullet_points(doc['key_points'])}\n\n"
                f"Technical Terms:\n{self.format_bullet_points(doc['technical_terms'])}\n\n"
                f"Most Relevant Content:\n{doc['text'][:1000]}...\n\n"
                f"Relationships:\n{self.format_bullet_points(doc['relationships'])}\n\n"
                f"Relevance Score: {doc['score']:.4f}\n"
                f"---\n"
            )
            llm_context.append(llm_section)
            
            user_metadata.append({
                "source": f"{doc['source_file']} ({doc['year']})",
                "summary": doc['summary'][:200] + "..." if doc['summary'] else "No summary available",
                "key_points": doc['key_points'],
                "relevance_score": doc['score'],
                "technical_terms": doc['technical_terms']
            })
        
        return "\n".join(llm_context), user_metadata

    @staticmethod
    def format_bullet_points(text: str, separator: str = ',') -> str:
        """Format text into bullet points."""
        if not text:
            return "Not available"
        points = [point.strip() for point in text.split(separator) if point.strip()]
        return '\n'.join(f"• {point}" for point in points)

    def pipe(self, user_message: str, model_id: str, messages: List[Dict[str, Any]], body: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """Main pipeline method for processing a query."""
        try:
            logger.info(f"Processing query: {user_message}")
            start_time = time.time()

            embedding = self.get_embedding(user_message)
            if not embedding:
                return "Failed to generate embedding for your question."
            
            documents = self.search_documents(embedding)
            if not documents:
                return "No relevant academic papers found for your question."
            
            context, metadata = self.format_context(documents)
            messages = [
                {"role": "system", "content": self.valves.SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Based on these academic papers:\n\n{context}\n\n"
                    f"Please answer this question: {user_message}\n\n"
                    "Remember to include year citations and structure your response "
                    "with clear sections as specified."
                )}
            ]
            
            processing_time = time.time() - start_time
            body["messages"] = messages
            body["metadata"] = {
                "sources": metadata,
                "query_timestamp": time.time(),
                "total_sources": len(documents),
                "average_score": sum(d['score'] for d in documents) / len(documents),
                "processing_time": f"{processing_time:.2f}s"
            }

            logger.info(f"Pipeline completed in {processing_time:.2f}s with {len(documents)} sources")
            return body
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        try:
            if hasattr(self, '_collection'):
                self._collection.release()
            connections.disconnect("default")
            logger.info("Successfully cleaned up resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")