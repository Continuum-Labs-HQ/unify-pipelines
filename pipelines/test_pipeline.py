import asyncio
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
import logging

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
            "What are the key developments in studying exoplanets and their atmospheres?",
        ]

    async def test_connectivity(self) -> bool:
        """Test connections to Milvus and the embedding service."""
        console.print("\n[bold cyan]Testing Service Connections[/bold cyan]")

        try:
            # Start pipeline (Milvus connection)
            await self.pipeline.on_startup()
            
            # Validate collection statistics
            if not self.pipeline._collection:
                raise ValueError("Milvus collection not loaded")
            
            entities = self.pipeline._collection.num_entities
            stats_table = Table(title="Collection Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            stats_table.add_row("Total Documents", str(entities))
            stats_table.add_row("Collection Name", self.pipeline.valves.MILVUS_COLLECTION)
            console.print(stats_table)
            
            # Test embedding generation
            test_text = "This is a test query for embedding generation"
            embedding = self.pipeline.get_embedding(test_text)
            if embedding and len(embedding) > 0:
                console.print(f"[green]✓ Embedding service working[/green]")
                return True
            else:
                raise ValueError("Failed to generate embedding")
        except Exception as e:
            console.print(f"[red]✗ Connectivity test failed: {str(e)}[/red]")
            return False

    async def test_search(self, query: str) -> Optional[List[Dict]]:
        """Test the Milvus search functionality."""
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
            
            # Display results in a table
            results_table = Table(title="Search Results")
            results_table.add_column("Source", style="cyan")
            results_table.add_column("Score", style="yellow")
            results_table.add_column("Summary", style="green")
            
            for doc in documents:
                results_table.add_row(
                    doc.get('source_file', 'Unknown'),
                    f"{doc.get('score', 0.0):.4f}",
                    doc.get('summary', 'N/A')[:100] + "..."
                )
            
            console.print(results_table)
            return documents
        except Exception as e:
            console.print(f"[red]✗ Search failed: {str(e)}[/red]")
            return None

    async def run_tests(self):
        """Execute the test suite."""
        try:
            # Test connectivity
            if not await self.test_connectivity():
                return
            
            # Test the first query's search functionality
            documents = await self.test_search(self.test_queries[0])
            if not documents:
                return
            
            # Test the full pipeline for all queries
            console.print("\n[bold cyan]Testing Full Pipeline[/bold cyan]")
            for query in self.test_queries:
                try:
                    result = self.pipeline.pipe(
                        user_message=query,
                        model_id="test_model",
                        messages=[],
                        body={"temperature": 0.7}
                    )
                    
                    if isinstance(result, dict) or isinstance(result, str):
                        console.print(f"[green]✓ Pipeline processed query: {query}[/green]")
                    else:
                        console.print(f"[red]✗ Pipeline failed for query: {result}[/red]")
                except Exception as e:
                    console.print(f"[red]✗ Query failed: {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]Error in test suite: {str(e)}[/red]")
        finally:
            # Cleanup resources
            await self.cleanup()

    async def cleanup(self):
        """Clean up pipeline resources."""
        console.print("\n[bold cyan]Cleaning up resources[/bold cyan]")
        try:
            await self.pipeline.on_shutdown()
        except Exception as e:
            console.print(f"[red]Error during cleanup: {str(e)}[/red]")


if __name__ == "__main__":
    console.print("[bold cyan]Starting Pipeline Tests[/bold cyan]")
    tester = PipelineTester()
    asyncio.run(tester.run_tests())
