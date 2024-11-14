"""
title: Academic RAG Pipeline
author: Assistant
date: 2024-11-14
version: 1.0
license: MIT
description: RAG pipeline for academic paper search using Milvus
requirements: pymilvus, requests, pydantic
"""

from typing import List, Union, Generator, Iterator, Dict, Any
from pydantic import BaseModel
import requests
from pymilvus import connections, Collection, utility
import logging

class Pipeline:
    class Valves(BaseModel):
        # Required base configuration
        pipelines: List[str] = ["*"]  # Which pipelines to connect to
        priority: int = 0             # Processing priority

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
        
        class Config:
            json_schema_extra = {
                "title": "Academic RAG Configuration",
                "description": "Configuration for ARXIV document search"
            }

    def __init__(self):
        self.name = "Academic RAG Pipeline"
        self.type = "processor"  # Added pipeline type
        self.valves = self.Valves()
        self._collection = None
        print(f"Initialized {self.name}")

    async def on_startup(self):
        """Initialization logic"""
        try:
            # Create connection
            connections.connect(
                alias="default",
                host=self.valves.MILVUS_HOST,
                port=self.valves.MILVUS_PORT,
                user=self.valves.MILVUS_USER,
                password=self.valves.MILVUS_PASSWORD
            )
            
            # Initialize collection
            self._collection = Collection(self.valves.MILVUS_COLLECTION)
            self._collection.load()
            print(f"Started {self.name}")
            
        except Exception as e:
            print(f"Startup error: {str(e)}")
            raise

    async def on_shutdown(self):
        """Cleanup logic"""
        try:
            if self._collection:
                self._collection.release()
            connections.disconnect("default")
            print(f"Shut down {self.name}")
        except Exception as e:
            print(f"Shutdown error: {str(e)}")

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Core processing logic"""
        try:
            # Handle title request
            if body.get("title", False):
                return self.name

            # Get embeddings
            response = requests.post(
                self.valves.EMBEDDING_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json={
                    "input": [user_message],
                    "model": self.valves.EMBEDDING_MODEL,
                    "input_type": "query",
                }
            )
            embedding = response.json()["data"][0]["embedding"]

            # Search documents
            results = self._collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 16}},
                limit=self.valves.TOP_K,
                output_fields=["source_file", "abstract", "key_points"]
            )

            # Process results
            documents = []
            for hits in results:
                for hit in hits:
                    if hit.distance <= self.valves.SCORE_THRESHOLD:
                        doc = {
                            "source_file": getattr(hit.entity, "source_file", "Unknown"),
                            "abstract": getattr(hit.entity, "abstract", "No abstract"),
                            "key_points": getattr(hit.entity, "key_points", "No key points"),
                            "score": float(hit.distance)
                        }
                        documents.append(doc)

            if not documents:
                return "No relevant documents found."

            # Format context
            context = "\n\n".join([
                f"Document {i+1}:\n"
                f"Source: {doc['source_file']}\n"
                f"Abstract: {doc['abstract']}\n"
                f"Key Points: {doc['key_points']}\n"
                f"Score: {doc['score']:.2f}\n"
                "---"
                for i, doc in enumerate(documents)
            ])

            # Update messages
            messages = [
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ]
            
            body["messages"] = messages
            return body
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return f"An error occurred: {str(e)}"