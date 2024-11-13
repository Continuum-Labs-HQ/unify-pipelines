from typing import List, Union, Generator, Iterator
import subprocess
import sys


class Pipeline:
    def __init__(self):
        self.name = "Text-to-Art Pipeline"

    async def on_startup(self):
        print(f"{self.name} is starting up...")

    async def on_shutdown(self):
        print(f"{self.name} is shutting down...")

    def execute_art_command(self, text: str, font: str) -> Union[str, None]:
        """
        Generate ASCII art for the given text using the 'art' library.
        """
        try:
            # Create the Python code dynamically
            code = f"""
from art import text2art
print(text2art("{text}", font="{font}"))
"""
            # Use the current Python executable to run the subprocess
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error generating ASCII art: {e.output.strip()}"

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process the user's text and generate ASCII art.
        """
        print(f"pipe: {__name__}")
        print(f"User message: {user_message}")

        # Extract font from the body or use a default font
        font = body.get("font", "block")
        if not user_message.strip():
            return "Error: Please provide some text to generate ASCII art."

        # Generate ASCII art
        ascii_art = self.execute_art_command(user_message.strip(), font)
        return ascii_art or "Failed to generate ASCII art."


# Example Usage
if __name__ == "__main__":
    import asyncio

    pipeline = Pipeline()
    asyncio.run(pipeline.on_startup())

    # Simulate user input
    user_message = "Hello, World!"
    body = {"font": "block"}  # Try different fonts like "block", "random", "thin", etc.
    output = pipeline.pipe(user_message, "", [], body)
    print(output)

    asyncio.run(pipeline.on_shutdown())
