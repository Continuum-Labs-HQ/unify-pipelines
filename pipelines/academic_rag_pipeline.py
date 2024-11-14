"""
title: Academic RAG Pipeline
author: Assistant
date: 2024-11-14
version: 1.0
description: Simple RAG pipeline for academic paper search using Milvus
requirements: pymilvus, requests, pydantic
"""

from typing import List, Union, Generator, Iterator, Dict, Any
from pydantic import BaseModel
import requests
from pymilvus import connections, Collection, utility
import logging

class Pipeline:
    class Valves(BaseModel):
        # Direct configuration for testing
        MILVUS_HOST: str = "10.106.175.99"
        MILVUS_PORT: str = "19530"
        MILVUS_USER: str = "thannon"
        MILVUS_PASSWORD: str = "chaeBio7!!!"
        MILVUS_COLLECTION: str = "arxiv_documents"
        EMBEDDING_ENDPOINT: str = "http://192.168.13.50:30000/v1/embeddings"
        EMBEDDING_MODEL: str = "nvidia/nv-embedqa-mistral-7b-v2"
        TOP_K: int = 5
        SCORE_THRESHOLD: float = 2.0

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
            self._collection.load()
            print("Connected to Milvus successfully")
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
        response = requests.post(
            self.valves.EMBEDDING_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json={
                "input": [text],
                "model": self.valves.EMBEDDING_MODEL,
                "input_type": "query",
            }
        )
        return response.json()["data"][0]["embedding"]

    def search_documents(self, embedding: List[float]) -> List[Dict[str, Any]]:
        """Search Milvus for relevant documents"""
        results = self._collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 16}},
            limit=self.valves.TOP_K,
            output_fields=["doc_id", "source_file", "text", "summary", "abstract"]
        )

        documents = []
        for hits in results:
            for hit in hits:
                if hit.distance <= self.valves.SCORE_THRESHOLD:
                    doc = {field: getattr(hit.entity, field, "") 
                          for field in hit.entity._row_data.keys()}
                    doc["score"] = float(hit.distance)
                    documents.append(doc)
        return documents

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

            # Get embeddings and search
            embedding = self.get_embedding(user_message)
            documents = self.search_documents(embedding)
            
            if not documents:
                return "No relevant documents found."

            # Format simple context
            context = "\n\n".join([
                f"Document {i+1}:\n{doc['abstract']}\n---"
                for i, doc in enumerate(documents)
            ])

            # Update messages for LLM
            messages = [
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
            ]
            
            body["messages"] = messages
            return body
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"An error occurred: {str(e)}"