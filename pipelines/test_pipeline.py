import asyncio
import time
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os
import logging
from pymilvus import connections

# Import the Pipeline class
from academic_rag_pipeline import Pipeline

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineTester:
    def __init__(self):
        self.pipeline = Pipeline()
        self.test_queries = [
            "What are the latest discoveries about black holes?",
            "How do different star systems form and evolve over time?",
            "What are the current challenges in exploring deep space?",
            "Explain the relationship between a star's mass and its life cycle.",
            "What are the key developments in studying exoplanets and their atmospheres?"
        ]

    async def test_connectivity(self) -> bool:
        """Test connections to services."""
        console.print("\n[bold cyan]Testing Service Connections[/bold cyan]")

        try:
            # Collection validation
            entities = self.pipeline._collection.num_entities
            
            stats_table = Table(title="Collection Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            stats_table.add_row("Total Documents", str(entities))
            stats_table.add_row("Collection Name", self.pipeline.valves.MILVUS_COLLECTION)
            console.print(stats_table)
            
            # Test embedding
            test_text = "This is a test query for embedding generation"
            embedding = self.pipeline.get_embedding(test_text)
            if embedding and len(embedding) == 4096:
                console.print(f"[green]✓ Embedding service working[/green]")
                return True
            
        except Exception as e:
            console.print(f"[red]✗ Connection test failed: {str(e)}[/red]")
            return False

    async def test_search(self, query: str) -> Optional[List[Dict]]:
        """Test search functionality."""
        console.print(f"\n[bold cyan]Testing Search[/bold cyan]")
        console.print(f"Query: [yellow]{query}[/yellow]")
        
        try:
            # Generate embedding and search
            embedding = self.pipeline.get_embedding(query)
            if not embedding:
                raise ValueError("Failed to generate embedding")
            
            documents = self.pipeline.search_documents(embedding)
            if not documents:
                raise ValueError("No documents found")
            
            # Display results
            results_table = Table(title="Search Results")
            results_table.add_column("Source")
            results_table.add_column("Score")
            results_table.add_column("Summary")
            
            for doc in documents:
                results_table.add_row(
                    doc['source_file'],
                    f"{doc['score']:.4f}",
                    f"{doc['summary'][:100]}..." if doc['summary'] else "N/A"
                )
            
            console.print(results_table)
            return documents
            
        except Exception as e:
            console.print(f"[red]✗ Search failed: {str(e)}[/red]")
            return None

    async def run_tests(self):
        """Execute test suite."""
        try:
            if not await self.test_connectivity():
                return
            
            # Test first query
            documents = await self.test_search(self.test_queries[0])
            if not documents:
                return
            
            # Test full pipeline
            console.print("\n[bold cyan]Testing Full Pipeline[/bold cyan]")
            for query in self.test_queries:
                try:
                    result = self.pipeline.pipe(
                        user_message=query,
                        model_id="test_model",
                        messages=[],
                        body={"temperature": 0.7}
                    )
                    
                    if isinstance(result, dict):
                        console.print(f"\n[green]✓ Pipeline processed: {query}[/green]")
                    else:
                        console.print(f"[red]✗ Pipeline failed: {result}[/red]")
                        
                except Exception as e:
                    console.print(f"[red]✗ Query failed: {str(e)}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error in test suite: {str(e)}[/red]")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.pipeline.cleanup()
        except:
            pass

if __name__ == "__main__":
    console.print("[bold cyan]Starting Pipeline Tests[/bold cyan]")
    tester = PipelineTester()
    asyncio.run(tester.run_tests())