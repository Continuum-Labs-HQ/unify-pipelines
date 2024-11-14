"""
title: Academic RAG Filter
author: Assistant
date: 2024-11-14
version: 1.0
license: MIT
description: Filter for academic paper search and context injection
requirements: pymilvus, requests, pydantic
"""

from typing import List, Union, Generator, Iterator, Dict, Any, Optional
from pydantic import BaseModel
import requests
from pymilvus import connections, Collection, utility
import logging

class Pipeline:
    """Filter pipeline for academic paper search and context injection"""
    
    class Valves(BaseModel):
        """Configuration parameters for the filter"""
        # Required filter configuration
        pipelines: List[str] = ["*"]
        priority: int = 0
        
        # Milvus configuration
        milvus_host: str = "10.106.175.99"
        milvus_port: str = "19530"
        milvus_user: str = "thannon"
        milvus_password: str = "chaeBio7!!!"
        milvus_collection: str = "arxiv_documents"
        
        # Embedding configuration
        embedding_endpoint: str = "http://192.168.13.50:30000/v1/embeddings"
        embedding_model: str = "nvidia/nv-embedqa-mistral-7b-v2"
        
        # Search configuration
        top_k: int = 5
        score_threshold: float = 2.0

        class Config:
            """Pydantic configuration"""
            json_schema_extra = {
                "title": "Academic RAG Filter Configuration",
                "description": "Settings for academic paper search and retrieval"
            }

    def __init__(self):
        """Initialize the filter pipeline"""
        self.type = "filter"
        self.name = "Academic RAG Filter"
        self.valves = self.Valves()
        self._collection = None
        logging.info(f"Initialized {self.name}")

    async def on_startup(self):
        """Server startup hook"""
        try:
            # Initialize Milvus connection
            connections.connect(
                alias="default",
                host=self.valves.milvus_host,
                port=self.valves.milvus_port,
                user=self.valves.milvus_user,
                password=self.valves.milvus_password
            )
            self._collection = Collection(self.valves.milvus_collection)
            self._collection.load()
            logging.info(f"Started {self.name}")
        except Exception as e:
            logging.error(f"Startup error: {str(e)}")

    async def on_shutdown(self):
        """Server shutdown hook"""
        try:
            if self._collection:
                self._collection.release()
            connections.disconnect("default")
            logging.info(f"Shut down {self.name}")
        except Exception as e:
            logging.error(f"Shutdown error: {str(e)}")

    def search_papers(self, query: str) -> Dict[str, Any]:
        """Search for relevant papers using the query"""
        try:
            # Get embedding
            response = requests.post(
                self.valves.embedding_endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "input": [query],
                    "model": self.valves.embedding_model,
                    "input_type": "query",
                }
            )
            embedding = response.json()["data"][0]["embedding"]

            # Search Milvus
            results = self._collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 16}},
                limit=self.valves.top_k,
                output_fields=["source_file", "abstract", "key_points"]
            )

            # Process results
            papers = []
            for hits in results:
                for hit in hits:
                    if hit.distance <= self.valves.score_threshold:
                        papers.append({
                            "source": getattr(hit.entity, "source_file", "Unknown"),
                            "abstract": getattr(hit.entity, "abstract", "No abstract"),
                            "key_points": getattr(hit.entity, "key_points", "No key points"),
                            "score": float(hit.distance)
                        })

            return {"success": True, "papers": papers}

        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def inlet(self, body: Dict, user: Optional[Dict] = None) -> Dict:
        """Process incoming messages"""
        try:
            # Skip if not a proper message body
            if not isinstance(body, dict) or "messages" not in body:
                return body

            # Get the last user message
            last_message = body["messages"][-1]["content"]
            
            # Search for relevant papers
            result = self.search_papers(last_message)
            
            if not result["success"]:
                logging.error(f"Search failed: {result.get('error')}")
                return body

            papers = result["papers"]
            if not papers:
                return body

            # Format context
            context = "\n\n".join([
                f"Document {i+1}:\n"
                f"Source: {paper['source']}\n"
                f"Abstract: {paper['abstract']}\n"
                f"Key Points: {paper['key_points']}\n"
                f"Relevance: {paper['score']:.2f}\n"
                "---"
                for i, paper in enumerate(papers)
            ])

            # Add context to system message
            system_msg = {
                "role": "system",
                "content": (
                    "You are a research assistant. Use these academic papers as context "
                    f"for your response:\n\n{context}"
                )
            }
            
            body["messages"].insert(0, system_msg)
            return body

        except Exception as e:
            logging.error(f"Inlet error: {str(e)}")
            return body

    async def outlet(self, response: str, user: Optional[Dict] = None) -> str:
        """Process outgoing messages"""
        return response
