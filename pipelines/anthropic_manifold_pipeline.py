"""
title: Aleph Manifold Pipeline
author: Your Name
date: 2024-01-20
version: 1.1
license: MIT
description: A pipeline for text generation and image analysis using the Anthropic API.
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator, Optional, Dict, Any
from pydantic import BaseModel
import sseclient
import logging
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""
        MAX_IMAGE_SIZE: int = 100 * 1024 * 1024  # 100MB limit
        MAX_IMAGES_PER_REQUEST: int = 5

    def __init__(self):
        """Initialize the Aleph pipeline."""
        self.type = "manifold"
        self.id = "aleph"
        self.name = "aleph"
        
        # Define model mapping
        self.model_mapping = {
            "1": "claude-3-5-sonnet-20241022",
            "2": "claude-3-sonnet-20240229",
            "3": "claude-3-opus-20240229",
            "4": "claude-3-haiku-20240307"
        }
        
        # Initialize valves with API key from environment
        self.valves = self.Valves(
            **{
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                "MAX_IMAGE_SIZE": int(os.getenv("MAX_IMAGE_SIZE", 100 * 1024 * 1024)),
                "MAX_IMAGES_PER_REQUEST": int(os.getenv("MAX_IMAGES_PER_REQUEST", 5))
            }
        )
        self.url = 'https://api.anthropic.com/v1/messages'
        self.update_headers()
        logger.info("Aleph pipeline initialized")

    def update_headers(self):
        """Update request headers with current API key."""
        self.headers = {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': self.valves.ANTHROPIC_API_KEY
        }
        logger.debug("Headers updated")

    def get_available_models(self):
        """Return list of available models with custom names."""
        return [
            {"id": model_id, "name": f"Aleph {model_id}"} 
            for model_id in self.model_mapping.keys()
        ]

    def validate_image_size(self, image_data: str) -> bool:
        """Validate image size is within limits."""
        try:
            if image_data.startswith('data:image'):
                _, b64_data = image_data.split(',', 1)
                size = len(b64_data) * 3 / 4
            else:
                response = requests.head(image_data, allow_redirects=True)
                size = int(response.headers.get('content-length', 0))
            
            return size <= self.valves.MAX_IMAGE_SIZE
        except Exception as e:
            logger.error(f"Error validating image size: {str(e)}")
            return False

    def process_image(self, image_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate image data for the Anthropic API."""
        try:
            if not isinstance(image_data, dict) or "url" not in image_data:
                logger.error("Invalid image data format")
                return None

            url = image_data["url"]
            if not self.validate_image_size(url):
                raise ValueError("Image size exceeds maximum allowed size")

            if url.startswith("data:image"):
                mime_type, base64_data = url.split(",", 1)
                media_type = mime_type.split(":")[1].split(";")[0]
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            else:
                if not url.startswith(('http://', 'https://')):
                    raise ValueError("Invalid image URL scheme")
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url
                    }
                }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None

    async def on_startup(self):
        """Handle startup tasks."""
        logger.info(f"Starting {self.name} pipeline")
        if not self.valves.ANTHROPIC_API_KEY:
            logger.warning("No Anthropic API key provided")

    async def on_shutdown(self):
        """Handle shutdown tasks."""
        logger.info(f"Shutting down {self.name} pipeline")

    async def on_valves_updated(self):
        """Handle valve updates."""
        self.update_headers()
        logger.info("Valves updated")

    def pipelines(self) -> List[dict]:
        """Return available pipelines (models)."""
        return self.get_available_models()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Process a message through the Anthropic API."""
        try:
            # Map our custom model name to Anthropic model ID
            anthropic_model_id = self.model_mapping.get(model_id)
            if not anthropic_model_id:
                raise ValueError(f"Unknown model: {model_id}")
                
            # Clean up body
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            # Process messages
            processed_messages = []
            image_count = 0

            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    processed_messages.append({
                        "role": message["role"],
                        "content": [{"type": "text", "text": content}]
                    })
                elif isinstance(content, list):
                    processed_content = []
                    for item in content:
                        if item.get("type") == "text":
                            processed_content.append({
                                "type": "text",
                                "text": item["text"]
                            })
                        elif item.get("type") == "image_url":
                            if image_count >= self.valves.MAX_IMAGES_PER_REQUEST:
                                logger.warning("Maximum images per request exceeded")
                                continue
                            processed_image = self.process_image(item["image_url"])
                            if processed_image:
                                processed_content.append(processed_image)
                                image_count += 1

                    if processed_content:
                        processed_messages.append({
                            "role": message["role"],
                            "content": processed_content
                        })

            # Prepare payload
            payload = {
                "model": anthropic_model_id,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.7),
                "stream": body.get("stream", False)
            }

            logger.info(f"Processing request with model: Aleph {model_id} (Anthropic: {anthropic_model_id})")
            
            # Handle streaming vs non-streaming
            if body.get("stream", False):
                return self.stream_response(payload)
            else:
                return self.get_completion(payload)

        except Exception as e:
            logger.error(f"Error in pipe: {str(e)}")
            return f"Error: {str(e)}"

    def stream_response(self, payload: dict) -> Generator:
        """Handle streaming responses from Anthropic API."""
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()

            client = sseclient.SSEClient(response)
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    if data["type"] == "content_block_start":
                        yield data["content_block"]["text"]
                    elif data["type"] == "content_block_delta":
                        yield data["delta"]["text"]
                    elif data["type"] == "message_stop":
                        break
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {event.data}")
                except KeyError as e:
                    logger.error(f"Unexpected data structure: {e}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming error: {str(e)}")
            raise Exception(f"Streaming error: {str(e)}")

    def get_completion(self, payload: dict) -> str:
        """Handle non-streaming responses from Anthropic API."""
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"] if "content" in result and result["content"] else ""
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Completion error: {str(e)}")
            raise Exception(f"Completion error: {str(e)}")
