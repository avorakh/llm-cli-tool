# LLM CLI Tool

The tool accepts a prompt as a CLI argument and streams the response back to your terminal in real-time, displaying token counts and cost at the end of each call.

## Setup

Create and activate a virtual environment, then install the required libraries:

```sh
python3 -m venv venv
source venv/bin/activate
pip install openai anthropic python-dotenv click tiktoken tenacity
```

## Run And Test

Basic call with streaming:

```sh
python cli.py "What is tokenization?"
```

Try a different model:

```sh
python cli.py --model gpt-4o "Explain attention mechanisms"
```

Disable streaming and get the full response at once:

```sh
python cli.py --no-stream "List 3 LLM concepts"
```

Get metrics as JSON:

```sh
python cli.py --json-output "Explain RAG in one sentence"
```
