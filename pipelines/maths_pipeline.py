
Here’s a script that uses the Pipeline class to create a "Math Problem Solver" pipeline. It dynamically solves basic math problems provided as user input and provides step-by-step solutions.

Script: Math Problem Solver Pipeline
This pipeline can:

Parse a math expression provided by the user.
Evaluate the expression safely.
Return the solution along with the steps.
python
Copy code
import ast
import operator
from typing import List, Union, Generator, Iterator

class MathSolverPipeline:
    def __init__(self):
        self.name = "Math Problem Solver"

    async def on_startup(self):
        print(f"{self.name} is starting up...")

    async def on_shutdown(self):
        print(f"{self.name} is shutting down...")

    def evaluate_expression(self, expression: str) -> Union[str, dict]:
        """
        Safely evaluates a math expression and provides step-by-step details.
        Supports basic operations: +, -, *, /, **.
        """
        # Supported operators
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }

        def eval_node(node):
            if isinstance(node, ast.Num):  # <number>
                return node.n
            elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                left = eval_node(node.left)
                right = eval_node(node.right)
                return ops[type(node.op)](left, right)
            else:
                raise ValueError("Unsupported expression")

        # Parse the expression
        try:
            tree = ast.parse(expression, mode="eval")
            value = eval_node(tree.body)

            return {
                "expression": expression,
                "steps": f"Calculated result of: {expression}",
                "result": value,
            }
        except Exception as e:
            return {"error": str(e)}

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process the user's math expression and return the solution.
        """
        print(f"Processing message: {user_message}")
        result = self.evaluate_expression(user_message)
        if "error" in result:
            return f"Error: {result['error']}"
        else:
            return f"Expression: {result['expression']}\nResult: {result['result']}\nSteps: {result['steps']}"

# Example Usage
if __name__ == "__main__":
    import asyncio

    pipeline = MathSolverPipeline()
    asyncio.run(pipeline.on_startup())

    # Simulate user input
    user_input = "2 + 3 * (4 - 1) ** 2"
    output = pipeline.pipe(user_input, "", [], {})
    print(output)

    asyncio.run(pipeline.on_shutdown())
