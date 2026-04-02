import asyncio
import ast
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


def calculate(expression: str) -> str:
    """Evaluate a basic math expression (numbers and + - * / ** % parentheses)."""
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    )
    try:
        node = ast.parse(expression, mode="eval")
        if not all(isinstance(n, allowed_nodes) for n in ast.walk(node)):
            return "Error: unsupported expression."
        value = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, {})
        return str(value)
    except Exception as exc:
        return f"Error: {exc}"


def fetch_webpage(url: str, max_chars: int = 3000) -> str:
    """Fetch plain text from a public URL."""
    try:
        with urlopen(url, timeout=15) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            content = response.read().decode(charset, errors="replace")
            return content[:max_chars]
    except URLError as exc:
        return f"Error fetching URL: {exc}"
    except Exception as exc:
        return f"Error: {exc}"


def read_local_file(path: str, max_chars: int = 3000) -> str:
    """Read a text file from this project directory."""
    root = Path(__file__).resolve().parent
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        return "Error: file must be inside the autogen-agent directory."
    if not target.exists() or not target.is_file():
        return "Error: file not found."
    try:
        return target.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception as exc:
        return f"Error: {exc}"


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Create a .env file (see .env.example) or set the env var."
        )

    model = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")

    model_client = OpenAIChatCompletionClient(
        model=model,
        # If you want to reduce cost/latency, you can swap the model via AUTOGEN_MODEL.
    )

    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message=(
            "You are the planner agent. Break the user request into clear steps for other agents. "
            "Keep plans short and actionable."
        ),
    )

    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        tools=[calculate, fetch_webpage, read_local_file],
        reflect_on_tool_use=True,
        system_message=(
            "You are the researcher agent. Gather facts, constraints, and options needed to solve the task. "
            "Use tools when useful: calculate for math, fetch_webpage for public URLs, "
            "and read_local_file for local project files."
        ),
    )

    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message=(
            "You are the writer agent. Produce the final response for the user using planner and researcher outputs. "
            "End your final response with the word TERMINATE."
        ),
    )

    # Stop when writer signals completion or after a safe max number of turns.
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
    team = RoundRobinGroupChat([planner, researcher, writer], termination_condition=termination)

    print("AutoGen multi-agent team ready (planner -> researcher -> writer). Type 'exit' to quit.")
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        result = await team.run(task=user_text)

        # Grab the last text response (typically from writer).
        last_text = ""
        for msg in result.messages:
            if isinstance(msg, TextMessage):
                last_text = msg.content
        print(f"Team: {last_text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")

