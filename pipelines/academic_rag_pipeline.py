"""
title: Academic RAG Pipeline
author: Assistant
date: 2024-11-14
version: 1.0
license: MIT
description: RAG pipeline for academic paper search using Milvus
requirements: pymilvus, requests, pydantic
"""

from typing import List, Union, Generator, Iterator, Dict, Any, Optional
from pydantic import BaseModel
import requests
from pymilvus import connections, Collection, utility
import logging

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0

        # Milvus Configuration
        MILVUS_HOST: str = "10.106.175.99"
        MILVUS_PORT: str = "19530"
        MILVUS_USER: str = "thannon"
        MILVUS_PASSWORD: str = "chaeBio7!!!"
        MILVUS_COLLECTION: str = "arxiv_documents"
        
        # Embedding Configuration
        EMBEDDING_ENDPOINT: str = "http://192.168.13.50:30000/v1/embeddings"
        EMBEDDING_MODEL: str = "nvidia/nv-embedqa-mistral-7b-v2"
        
        TOP_K: int = 5
        SCORE_THRESHOLD: float = 2.0

    def __init__(self):
        self.name = "Academic RAG Pipeline"
        self.type = "filter"  # Changed to filter type
        self.valves = self.Valves()
        self._collection = None
        self._connected = False

    def _connect_milvus(self):
        """Helper method to establish Milvus connection"""
        if not self._connected:
            connections.connect(
                alias="default",
                host=self.valves.MILVUS_HOST,
                port=self.valves.MILVUS_PORT,
                user=self.valves.MILVUS_USER,
                password=self.valves.MILVUS_PASSWORD
            )
            self._collection = Collection(self.valves.MILVUS_COLLECTION)
            self._collection.load()
            self._connected = True
            print("Connected to Milvus")

    async def on_startup(self):
        """Startup initialization"""
        try:
            self._connect_milvus()
            print(f"Started {self.name}")
        except Exception as e:
            print(f"Startup error: {str(e)}")
            self._connected = False

    async def on_shutdown(self):
        """Cleanup resources"""
        if self._connected:
            try:
                self._collection.release()
                connections.disconnect("default")
                self._connected = False
                print(f"Shut down {self.name}")
            except Exception as e:
                print(f"Shutdown error: {str(e)}")

    def process_message(self, user_message: str) -> Dict:
        """Core processing logic"""
        try:
            if not self._connected:
                self._connect_milvus()

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
                return {"error": "No relevant documents found."}

            context = "\n\n".join([
                f"Document {i+1}:\n"
                f"Source: {doc['source_file']}\n"
                f"Abstract: {doc['abstract']}\n"
                f"Key Points: {doc['key_points']}\n"
                f"Score: {doc['score']:.2f}\n"
                "---"
                for i, doc in enumerate(documents)
            ])

            return {"context": context}

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return {"error": str(e)}

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Pre-processing filter"""
        try:
            if not isinstance(body, dict):
                return body

            if "messages" not in body:
                return body

            last_message = body["messages"][-1]["content"]
            result = self.process_message(last_message)

            if "error" in result:
                return body

            context = result["context"]
            
            # Add context to the system message
            system_message = {
                "role": "system",
                "content": "You are a research assistant. Use the following academic context to answer the question:\n\n" + context
            }
            
            body["messages"].insert(0, system_message)
            return body

        except Exception as e:
            print(f"Inlet error: {str(e)}")
            return body

    async def outlet(self, response: str, user: Optional[dict] = None) -> str:
        """Post-processing filter"""
        return response

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipe method"""
        if body.get("title", False):
            return self.name
        return f"An error occurred: Pipeline should be used as a filter"
