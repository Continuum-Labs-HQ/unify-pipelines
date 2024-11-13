from text_to_art import Pipeline
import asyncio

def test_on_startup_shutdown(pipeline):
    """Test the lifecycle methods of the pipeline."""
    try:
        asyncio.run(pipeline.on_startup())
        print("[PASSED] on_startup test")
    except Exception as e:
        print(f"[FAILED] on_startup test: {e}")

    try:
        asyncio.run(pipeline.on_shutdown())
        print("[PASSED] on_shutdown test")
    except Exception as e:
        print(f"[FAILED] on_shutdown test: {e}")


def test_execute_art_command(pipeline):
    """Test the ASCII art generation using the art library."""
    text = "Test"
    font = "block"

    try:
        output = pipeline.execute_art_command(text, font)
        print("Generated ASCII Art:", output)  # Debug output
        assert "Test" in output, "ASCII art does not contain the expected text"
        assert len(output) > 0, "Output is empty"
        print("[PASSED] execute_art_command test")
    except AssertionError as e:
        print(f"[FAILED] execute_art_command test: {e}")
    except Exception as e:
        print(f"[FAILED] execute_art_command unexpected error: {e}")


def test_pipe_method(pipeline):
    """Test the pipeline's main processing method."""
    user_message = "Hello, ASCII!"
    body = {"font": "block"}

    try:
        output = pipeline.pipe(user_message, "", [], body)
        print("Pipeline Output:", output)  # Debug output
        assert "Hello" in output, "Pipeline output does not contain expected ASCII art"
        assert len(output) > 0, "Pipeline output is empty"
        print("[PASSED] pipe_method test")
    except AssertionError as e:
        print(f"[FAILED] pipe_method test: {e}")
    except Exception as e:
        print(f"[FAILED] pipe_method unexpected error: {e}")


def test_pipe_with_empty_message(pipeline):
    """Test the pipe method with an empty input message."""
    user_message = ""
    body = {"font": "block"}

    try:
        output = pipeline.pipe(user_message, "", [], body)
        assert output == "Error: Please provide some text to generate ASCII art.", \
            "Empty message did not return the expected error"
        print("[PASSED] pipe_with_empty_message test")
    except AssertionError as e:
        print(f"[FAILED] pipe_with_empty_message test: {e}")
    except Exception as e:
        print(f"[FAILED] pipe_with_empty_message unexpected error: {e}")


def test_pipe_with_invalid_font(pipeline):
    """Test the pipe method with an invalid font."""
    user_message = "Invalid Font Test"
    body = {"font": "nonexistent_font"}

    try:
        output = pipeline.pipe(user_message, "", [], body)
        print("Pipeline Output (Invalid Font):", output)  # Debug output
        assert "Error generating ASCII art" in output or "Failed to generate ASCII art" in output, \
            "Invalid font did not return the expected error"
        print("[PASSED] pipe_with_invalid_font test")
    except AssertionError as e:
        print(f"[FAILED] pipe_with_invalid_font test: {e}")
    except Exception as e:
        print(f"[FAILED] pipe_with_invalid_font unexpected error: {e}")


if __name__ == "__main__":
    print("Starting Text-to-Art Pipeline Tests...\n")
    pipeline = Pipeline()

    # Run tests
    test_on_startup_shutdown(pipeline)
    test_execute_art_command(pipeline)
    test_pipe_method(pipeline)
    test_pipe_with_empty_message(pipeline)
    test_pipe_with_invalid_font(pipeline)

    print("\nAll tests completed.")
