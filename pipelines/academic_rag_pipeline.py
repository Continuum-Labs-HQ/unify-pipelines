"""
title: Academic RAG Pipeline
author: Assistant
date: 2024-11-14
version: 1.0
description: RAG pipeline aligned with Milvus schema for ARXIV documents
requirements: pymilvus, requests, pydantic
"""

from typing import List, Union, Generator, Iterator, Dict, Any
from pydantic import BaseModel
import requests
from pymilvus import connections, Collection, utility
import logging

class Pipeline:
    class Valves(BaseModel):
        # Milvus Configuration
        MILVUS_HOST: str = "10.106.175.99"
        MILVUS_PORT: str = "19530"
        MILVUS_USER: str = "thannon"
        MILVUS_PASSWORD: str = "chaeBio7!!!"
        MILVUS_COLLECTION: str = "arxiv_documents"
        
        # Embedding Configuration
        EMBEDDING_ENDPOINT: str = "http://192.168.13.50:30000/v1/embeddings"
        EMBEDDING_MODEL: str = "nvidia/nv-embedqa-mistral-7b-v2"
        
        # Search Configuration
        TOP_K: int = 5
        SCORE_THRESHOLD: float = 2.0
        
        # Schema-aligned output fields
        OUTPUT_FIELDS: List[str] = [
            "doc_id",
            "source_file",
            "text",
            "summary",
            "key_points",
            "technical_terms",
            "abstract",
            "relationships",
            "timestamp"
        ]

        class Config:
            json_schema_extra = {
                "title": "Academic RAG Configuration",
                "description": "Configuration for ARXIV document search"
            }

    def __init__(self):
        self.name = "Academic RAG Pipeline"
        self.valves = self.Valves()
        self._collection = None

    async def on_startup(self):
        """Initialize Milvus connection"""
        try:
            connections.connect(
                alias="default",
                host=self.valves.MILVUS_HOST,
                port=self.valves.MILVUS_PORT,
                user=self.valves.MILVUS_USER,
                password=self.valves.MILVUS_PASSWORD
            )
            self._collection = Collection(self.valves.MILVUS_COLLECTION)
            print(f"Connected to collection: {self._collection.name}")
            print(f"Number of entities: {self._collection.num_entities}")
            self._collection.load()
            
        except Exception as e:
            print(f"Startup error: {str(e)}")
            raise

    async def on_shutdown(self):
        """Cleanup resources"""
        if self._collection:
            self._collection.release()
        connections.disconnect("default")
        print("Pipeline shut down")

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings from endpoint"""
        try:
            response = requests.post(
                self.valves.EMBEDDING_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json={
                    "input": [text],
                    "model": self.valves.EMBEDDING_MODEL,
                    "input_type": "query",
                }
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            print(f"Generated embedding of size: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            raise

    def search_documents(self, embedding: List[float]) -> List[Dict[str, Any]]:
        """Schema-aligned search implementation"""
        try:
            search_params = {
                "metric_type": "L2",
                "params": {
                    "search_width": 128,
                    "nprobe": 16
                }
            }
            
            results = self._collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=self.valves.TOP_K,
                output_fields=self.valves.OUTPUT_FIELDS
            )

            documents = []
            for hits in results:
                for hit in hits:
                    if hit.distance <= self.valves.SCORE_THRESHOLD:
                        doc = {}
                        # Add the score
                        doc["score"] = float(hit.distance)
                        
                        # Safely extract all fields from the entity
                        for field in self.valves.OUTPUT_FIELDS:
                            value = getattr(hit.entity, field, None)
                            if value is not None:
                                doc[field] = str(value)
                        
                        documents.append(doc)
                        print(f"Found document: {doc['source_file']} with score: {doc['score']}")

            return documents

        except Exception as e:
            print(f"Search error: {str(e)}")
            raise

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline method"""
        try:
            if body.get("title", False):
                return self.name

            print(f"Processing query: {user_message}")

            # Get embeddings and search
            embedding = self.get_embedding(user_message)
            documents = self.search_documents(embedding)
            
            if not documents:
                return "No relevant documents found."

            # Format context with all available information
            context_parts = []
            for i, doc in enumerate(documents, 1):
                context_parts.append(
                    f"Document {i}:\n"
                    f"Source: {doc.get('source_file', 'Unknown')}\n"
                    f"Abstract: {doc.get('abstract', 'No abstract available')}\n"
                    f"Key Points: {doc.get('key_points', 'No key points available')}\n"
                    f"Technical Terms: {doc.get('technical_terms', 'No terms available')}\n"
                    f"Score: {doc.get('score', 0.0):.2f}\n"
                    "---"
                )

            context = "\n\n".join(context_parts)

            # Update messages for LLM
            messages = [
                {"role": "system", "content": "You are a research assistant with expertise in scientific papers."},
                {"role": "user", "content": f"Based on these papers:\n\n{context}\n\nQuestion: {user_message}"}
            ]
            
            body["messages"] = messages
            return body
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            return f"An error occurred: {str(e)}"